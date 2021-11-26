from train import *
from inference import *
from config_grid_search import *

if __name__ == "__main__":
    QUIET = True 
    SAVE_MODEL = False
    SAVE_TO_LOG = True
    DISTRIBUTED_TRAINING = False

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', 
        dest = 'gpu', nargs = '+', type = int, 
        default = 0, help = "GPU enable for running the procces"
    )

    args = parser.parse_args()
    GPU  = args.gpu[0]

    GLOBAL_LOGGER = GlobalLogger(
        path_to_global_logger = f'logs/stage-{STAGE}/gpu-{GPU}/logger_gpu_{GPU}.csv', 
        save_to_log = SAVE_TO_LOG
    )

    experiments_cfg = configs[f'GPU-{GPU}']
    # print(experiments_cfg)

    counter = 0
    PATH_TO_MODELS_ORIGINAL = copy.deepcopy(PATH_TO_MODELS)
    no_experiments = np.prod([len(val) for key, val in experiments_cfg.items()])
    # print(f"No. Experiments: {no_experiments}")

    for model_cfg in experiments_cfg['model_config']:
        for optimizer in experiments_cfg['optimizers']:
            for schedulers in experiments_cfg['schedulers']:
                for augmentation in experiments_cfg['augmentations']:
                    for learning_rate in experiments_cfg['lrs']:
                        PATH_TO_MODELS = copy.deepcopy(PATH_TO_MODELS_ORIGINAL)
                        counter += 1

                        CFG = {
                            'id': GLOBAL_LOGGER.get_version_id(),

                            'model_name': 'swin_large_patch4_window12_384_in22k', # 'beit_large_patch16_224_in22k', # 'swin_large_patch4_window12_384_in22k',
                            'dropout': 0.5,
                            'size': 384,
                            'batch_size_t': 4,
                            'batch_size_v': 32,

                            # Criterion and Gradient Control
                            'n_targets': 5,
                            'criterion': "CrossEntropyLoss",
                            'gradient_accumulation_steps': 1,
                            'max_gradient_norm': None,

                            # Parameters for optimizers, schedulers and learning_rate
                            'optimizer': "RangerLars",
                            'scheduler': "CosineAnnealingWarmRestarts",
                            
                            'LR': 0.00001,
                            'T_0': 200,
                            'T_max': 10,
                            'T_mult': 2,
                            'min_lr': 1e-6,
                            'max_lr': 1e-4,
                            'no_batches': 'NA',
                            'warmup_epochs': 1,
                            'cosine_epochs': 19,
                            'epochs' : 5,
                            'update_per_batch': True,

                            'num_workers': 4,
                            'n_folds': 5,

                            # Augumentations and other obserrvations for experiment
                            'train_transforms': [],
                            'valid_transforms': [],
                            'observations': None, # "Removing 'very_hard' Samples",

                            # Stochastic Weight Averaging
                            'use_swa': True,
                            'swa_lr':  0.01,
                            'swa_epoch':  0,

                            # Adaptive Sharpness-Aware Minimization
                            'use_sam':  True,
                            'use_asam': True,
                            'asam_rho': 0.1,

                            # Parameters for script control
                            'print_freq': 50,
                            'one_fold': True,
                            'use_apex': True,
                            'distributed_training': DISTRIBUTED_TRAINING, # python -m torch.distributed.launch --nproc_per_node=1 train.py
                            'save_to_log': SAVE_TO_LOG
                        }


                        CFG['id']           = GLOBAL_LOGGER.get_version_id()
                        CFG['model_name']   = model_cfg[0]
                        CFG['size']         = model_cfg[1]
                        CFG['batch_size_t'] = optimizer[1][0]
                        CFG['batch_size_v'] = optimizer[1][1]

                        CFG['optimizer']    = optimizer[0]
                        CFG['LR']           = learning_rate

                        CFG['scheduler']    = schedulers['scheduler_name']

                        if schedulers['scheduler_name'] == 'CosineAnnealingWarmRestarts':
                            CFG['T_0']    = schedulers['T_0']
                            CFG['T_mult'] = schedulers['T_mult']
                            CFG['min_lr'] = schedulers['min_lr']
                        elif schedulers['scheduler_name'] == 'OneCycleLR':
                            CFG['no_batches'] = optimizer[2]
                            CFG['epochs']     = schedulers['epochs']
                            CFG['max_lr']     = learning_rate * 5

                        CFG['train_transforms']   = augmentation

                        if SAVE_TO_LOG:
                            PATH_TO_MODELS = os.path.join(PATH_TO_MODELS, 'gpu-{}/model_{}'.format(GPU, CFG['id']))
                            if os.path.isdir(PATH_TO_MODELS) == False: os.makedirs(PATH_TO_MODELS)
                            logger = Logger(os.path.join(PATH_TO_MODELS, 'model_{}.log'.format(CFG['id'])), distributed = QUIET)
                        else:
                            logger = Logger(distributed = QUIET)

                        accuracy, best_models = run(GPU, CFG, GLOBAL_LOGGER, PATH_TO_MODELS, logger)

                        inference(best_models, CFG, GPU, VERBOSE = (not QUIET))

                        if SAVE_MODEL:
                            for fold, (accuracy, model) in enumerate(best_models): 
                                torch.save(
                                    model, 
                                    os.path.join(PATH_TO_MODELS, f"model_{CFG['id']}_name_{CFG['model_name']}_fold_{fold}_accuracy_{accuracy:.2f}.pth")
                                )

                        print(f'[GPU-{GPU}] Experiment {counter}/{no_experiments} -> Accuracy: {accuracy}')
                        logger.close()

                        free_gpu_memory(device = torch.device(f'cuda:{GPU}'))
