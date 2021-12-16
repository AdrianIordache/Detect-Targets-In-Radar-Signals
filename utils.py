import os
import gc
import cv2
import glob
import copy
import math
import time
import pickle
import random
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from PIL import Image
from typing import Dict
from IPython.display import display

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributed as dist

from torch.optim import Adam as Adam
from torch.optim import AdamW as AdamW
from torch.cuda.amp import autocast, GradScaler
from torchtools.optim import Ranger, RangerLars, Novograd, Ralamb
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from warmup_scheduler import GradualWarmupScheduler
from torch.optim.lr_scheduler import CosineAnnealingLR, CosineAnnealingWarmRestarts, OneCycleLR
from torch.distributed import Backend

from torchvision import datasets, transforms
from albumentations.pytorch import ToTensorV2
import albumentations as A
import timm

from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from asam import SAM
from torch.optim.swa_utils import AveragedModel, SWALR

import warnings
warnings.filterwarnings("ignore")

RANK = 0
WORLD_SIZE = torch.cuda.device_count()
DEVICE = torch.device('cuda:{}'.format(RANK) if torch.cuda.is_available() else 'cpu')

DECIMALS = 4
VERBOSITY = 30
RD = lambda x: np.round(x, DECIMALS)

def seed_everything(SEED = 42):
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    generator = torch.Generator()
    generator.manual_seed(SEED)

def seed_worker(worker_id):
    np.random.seed(SEED)
    random.seed(SEED)

OUTPUT = {
    "oof-accuracy":  0,
    "oof-precision": 0,
    "oof-recall":    0,

    "cross-validation"   : None,
    "public-leaderboard" : 0
}

SEED = 42
seed_everything(SEED)

STAGE = 3
PATH_TO_DATA       = 'data/detect-targets-in-radar-signals/'
PATH_TO_OOF        = 'logs/stage-{}/gpu-{}/oof.csv'.format(STAGE, RANK)
PATH_TO_MODELS     = 'models/stage-{}/'.format(STAGE)
PATH_TO_EMBEDDINGS = 'embeddings/'

PATH_TO_TRAIN_IMAGES = os.path.join(PATH_TO_DATA, "train/")
PATH_TO_TRAIN_META   = os.path.join(PATH_TO_DATA, "train.csv") 
PATH_TO_TRAIN_FE     = os.path.join(PATH_TO_DATA, "train_fe.csv")

PATH_TO_TEST_IMAGES  = os.path.join(PATH_TO_DATA, "test/")
PATH_TO_TEST_META    = os.path.join(PATH_TO_DATA, "test.csv")
PATH_TO_TEST_FE      = os.path.join(PATH_TO_DATA, "test_fe.csv")

PATH_TO_SUBMISSION   = os.path.join(PATH_TO_DATA, "sample_submission.csv")
PATH_TO_DENOISING    = os.path.join(PATH_TO_DATA, 'denoising', 'demo', 'denoising.csv')

PATH_TO_ERROR_ANALYSIS = os.path.join(PATH_TO_DATA, "error_analysis.csv")

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class GlobalLogger:
    def __init__(self, path_to_global_logger: str, save_to_log: bool):
        self.save_to_log = save_to_log
        self.path_to_global_logger = path_to_global_logger

        if os.path.exists(self.path_to_global_logger):
            self.logger = pd.read_csv(self.path_to_global_logger)

    def append(self, config_file: Dict, output_file: Dict):
        if self.save_to_log == False: return

        if os.path.exists(self.path_to_global_logger) == False:
            config_columns = [key for key in config_file.keys()]
            output_columns = [key for key in output_file.keys()]

            columns = config_columns + output_columns 
            logger = pd.DataFrame(columns = columns)
            logger.to_csv(self.path_to_global_logger, index = False)
            
        self.logger = pd.read_csv(self.path_to_global_logger)
        sample = {**config_file, **output_file}
        columns = [key for (key, value) in sample.items()]

        row = [value for (key, value) in sample.items()]
        row = np.array(row)
        row = np.expand_dims(row, axis = 0)

        sample = pd.DataFrame(row, columns = columns)
        self.logger = self.logger.append(sample, ignore_index = True)
        self.logger.to_csv(self.path_to_global_logger, index = False)

    
    def get_version_id(self):
        if os.path.exists(self.path_to_global_logger) == False: return 0
        logger = pd.read_csv(self.path_to_global_logger)
        ids = logger["id"].values
        if len(ids) == 0: return 0
        return ids[-1] + 1
    
    def view(self):
        from IPython.display import display
        display(self.logger)


class Logger:
    def __init__(self, path_to_logger: str = 'logger.log', distributed = False):
        from logging import getLogger, INFO, FileHandler,  Formatter,  StreamHandler

        self.logger = getLogger(__name__)
        self.logger.setLevel(INFO)

        if distributed == False:
            handler1 = StreamHandler()
            handler1.setFormatter(Formatter("%(message)s"))
            self.logger.addHandler(handler1)

        handler2 = FileHandler(filename = path_to_logger)
        handler2.setFormatter(Formatter("%(message)s"))
        self.logger.addHandler(handler2)

    def print(self, message):
        self.logger.info(message)

    def close(self):
        handlers = self.logger.handlers[:]
        for handler in handlers:
            handler.close()
            self.logger.removeHandler(handler)

def generate_folds(data: pd.DataFrame, skf_column: str, n_folds: int = 5, random_state = SEED):
    skf = StratifiedKFold(n_splits = n_folds, shuffle = True, random_state = SEED)
    for fold, (train_idx, valid_idx) in enumerate(skf.split(data, data[skf_column])):
        data.loc[valid_idx, 'fold'] = fold

    data['fold'] = data['fold'].astype(int)
    return data

def get_optimizer(parameters, config_file):
    assert config_file["optimizer"] in \
        ["Adam", "AdamW", "RangerLars", "Ranger", "Novograd", "Ralamb"], "[optimizer] -> Option Not Implemented"

    if config_file["optimizer"] == "Adam":
        return Adam(parameters, lr = config_file["LR"])
    if config_file["optimizer"] == "AdamW":
        return AdamW(parameters, lr = config_file["LR"])
    if config_file["optimizer"] == "RangerLars":
        return RangerLars(parameters, lr = config_file["LR"])
    if config_file["optimizer"] == "Ranger":
        return Ranger(parameters, lr = config_file["LR"])
    if config_file["optimizer"] == "Novograd":
        return Novograd(parameters, lr = config_file["LR"])
    if config_file["optimizer"] == "Ralamb":
        return Ralamb(parameters, lr = config_file["LR"])

    return None

def get_scheduler(optimizer, scheduler_params = None):
    if scheduler_params['scheduler'] == 'CosineAnnealingWarmRestarts':
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0              = scheduler_params['T_0'],
            T_mult           = scheduler_params['T_mult'],
            eta_min          = scheduler_params['min_lr'],
            last_epoch       = -1,
        )
    elif scheduler_params['scheduler'] == 'OneCycleLR':
        scheduler = OneCycleLR(
            optimizer,
            max_lr          = scheduler_params['max_lr'],
            steps_per_epoch = scheduler_params['no_batches'],
            epochs          = scheduler_params['epochs'],
        )
    elif scheduler_params['scheduler'] == 'CosineAnnealingLR':
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max          = scheduler_params['T_max'],
            eta_min        = scheduler_params['min_lr'],
            last_epoch     = -1
        )

    return scheduler

def get_criterion(config_file):
    if config_file["criterion"] == "L1Loss":
        return nn.L1Loss(reduction = 'mean')
    if config_file["criterion"] == "MSELoss":
        return nn.MSELoss(reduction = 'mean')
    if config_file['criterion'] == "RMSELoss":
        return RMSELoss()
    if config_file['criterion'] == "BCEWithLogitsLoss":
        return nn.BCEWithLogitsLoss()
    if config_file['criterion'] == "CrossEntropyLoss":
        return nn.CrossEntropyLoss()

    return None

def free_gpu_memory(device, object = None, verbose = False):
    if object == None:
        for object in gc.get_objects():
            try:
                if torch.is_tensor(object) or (hasattr(object, 'data' and  torch.is_tensor(object.data))):
                    if verbose: print(f"GPU Memory Used: {object}, with size: {object.size()}")
                    object = object.detach().cpu()
                    del object
            except:
                pass
    else:
        object = object.detach().cpu()
        del object

    gc.collect()
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

def create_denoising_dataset(path_to_clear, path_to_noisy, path_to_output):
    ids, labels, paths = [], [], []
    for path in [path_to_clear, path_to_noisy]:
        for image_path in sorted(glob.glob(path + '/*.png')):
            idx   = image_path.split('/')[-1]
            label = 1 if "noisy" in path else 0

            ids.append(idx)
            labels.append(label)
            paths.append(image_path)
          
    dataset = pd.DataFrame({
        'id': ids,
        'label': labels,
        'path': paths    
    }).sample(
        frac = 1, 
        random_state = SEED
    )

    dataset.to_csv(os.path.join(path_to_output, 'denoising.csv'), index = False)
    
def rebuild_oof(PATH_TO_MODELS):
    GPUS   = [0, 1]
    MODELS = [0, 1, 2]
    FOLDS  = [0, 1, 2, 3, 4]

    counter = 0
    all_ids, all_labels, all_predictions = [], [], []
    for gpu in GPUS:
        for model in MODELS:
            path = os.path.join(PATH_TO_MODELS, f'gpu-{gpu}', f'model_{model}')
            name = f"Stage-0-GPU-{gpu}-Model-{model}"
            ids, labels, predictions = [], [], []
            for fold in FOLDS:
                for file in sorted(glob.glob(path + '/*.pth')):
                    if f"fold_{fold}" in file:
                        print("Current Model: {}".format(file))
                        states = torch.load(file, map_location = torch.device('cpu'))
                        
                        ids.extend(states['oof_ids'])
                        labels.extend(states['oof_labels'])
                        predictions.extend(states['oof_proba'])

            counter += 1
            if len(predictions) != 15500: 
                print(f"Predictions found: {len(predictions)}, skipped model: {name}")
                continue

            all_ids.append(ids)
            all_labels.append(labels)
            all_predictions.append((name, predictions))

    results = {
        "id": all_ids[0],
        "label": all_labels[0],
    }

    data = pd.DataFrame(results)

    for column, predictions in all_predictions:
        data[column] = predictions

    display(data)
    data.to_csv('oof_rebuild.csv', index = False)


if __name__ == "__main__":
    create_denoising_dataset(
        'data/detect-targets-in-radar-signals/denoising/demo/clear/',
        'data/detect-targets-in-radar-signals/denoising/demo/noisy/',
        'data/detect-targets-in-radar-signals/denoising/demo/'
    )
