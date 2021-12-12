from utils   import *
from models  import *
from dataset import *

def train_epoch(model, loader, optimizer, criterion, scheduler, epoch, device, scaler, CFG, logger):
    model.train()
    losses_plot = []
    scores_plot = []

    losses      = AverageMeter()
    accuracies  = AverageMeter()

    start = end = time.time()

    for batch, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        if CFG['use_apex']:
            with autocast():
                preds = model(images).squeeze(1)
                loss  = criterion(preds, labels.long())
        else:
            preds = model(images).squeeze(1)
            loss  = criterion(preds, labels.long())

        output = torch.argmax(preds, dim = 1)

        output  = output.detach().cpu().numpy()
        targets = labels.detach().cpu().numpy()

        accuracy = accuracy_score(output, targets)

        losses.update(loss.item(), CFG['batch_size_t'])
        accuracies.update(accuracy * 100, CFG['batch_size_t'])

        if CFG['gradient_accumulation_steps'] > 1:
            loss = loss / CFG['gradient_accumulation_steps']

        if CFG['use_apex']:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch + 1) % CFG['gradient_accumulation_steps'] == 0:
            if CFG['use_apex']:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()

            if CFG['update_per_batch']: scheduler.step()

            optimizer.zero_grad()

        end = time.time()

        if batch % CFG['print_freq'] == 0 or batch == (len(loader) - 1):
            logger.print('[GPU {0}][TRAIN] Epoch: [{1}][{2}/{3}], Elapsed {remain:s}, Accuracy: {accuracy.val:.3f}({accuracy.avg:.3f}), Loss: {loss.val:.3f}({loss.avg:.3f}), LR: {lr:.6f}'
                  .format(device, epoch + 1, batch, len(loader), remain = timeSince(start, float(batch + 1) / len(loader)), accuracy = accuracies,
                          loss = losses, lr = scheduler.get_lr()[0]))

        losses_plot.append(losses.val)
        scores_plot.append(accuracies.val)

    free_gpu_memory(device)
    return accuracies.avg, np.mean(losses_plot), np.mean(scores_plot)


def valid_epoch(model, loader, criterion, device, CFG, logger):
    model.eval()
    losses_plot = []
    scores_plot = []

    predictions = []
    losses      = AverageMeter()
    accuracies  = AverageMeter()

    start = end = time.time()
    for batch, (images, labels) in enumerate(loader):
        images = images.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            preds = model(images).squeeze(1)

        loss    = criterion(preds, labels.long())
        output  = torch.argmax(preds, dim = 1)

        output  = output.detach().cpu().numpy()
        labels  = labels.detach().cpu().numpy()

        accuracy = accuracy_score(output, labels)

        losses.update(loss.item(), CFG['batch_size_v'])
        accuracies.update(accuracy * 100, CFG['batch_size_v'])

        end = time.time()

        predictions.extend(copy.deepcopy(output))
        if batch % CFG['print_freq'] == 0 or batch == (len(loader) - 1):
            logger.print('[GPU {0}][VALID][{1}/{2}], Elapsed {remain:s}, Accuracy: {accuracy.val:.3f}({accuracy.avg:.3f}), Batch Loss: {loss.val:.4f}, Average Loss: {loss.avg:.4f}'
                  .format(device, batch, len(loader), remain = timeSince(start, float(batch + 1) / len(loader)), accuracy = accuracies, loss = losses))

        losses_plot.append(losses.val)
        scores_plot.append(accuracies.val)

    free_gpu_memory(device)
    return accuracies.avg, predictions, np.mean(losses_plot), np.mean(scores_plot)

def train_fold(CFG: Dict, data: pd.DataFrame, fold: int, oof: np.array, logger, PATH_TO_MODELS: str, DEVICE,  swa_oof: np.array = None):
    logger.print(50 * "=" + " Training Fold: {} ".format(fold) + 50 * "=")
    train_idx = data[data['fold'] != fold].index
    valid_idx = data[data['fold'] == fold].index

    train_df = data.loc[train_idx].reset_index(drop = True)
    valid_df = data.loc[valid_idx].reset_index(drop = True)

    valid_labels = valid_df['label'].values
    valid_ids    = valid_df['id'].values

    train_transforms = A.Compose( 
        CFG['train_transforms'] + [
        A.Resize(CFG['size'], CFG['size']),
        A.Normalize(mean = [0.485, 0.456, 0.406], std  = [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    valid_transforms = A.Compose(
        CFG['valid_transforms'] + [
        A.Resize(CFG['size'], CFG['size']),
        A.Normalize(mean = [0.485, 0.456, 0.406], std  = [0.229, 0.224, 0.225]),
        ToTensorV2()
    ])

    trainset = RadarSignalsDataset(
            train_df, 
            train         = True, 
            transform     = train_transforms, 
    )

    validset = RadarSignalsDataset(
            valid_df, 
            train         = True, 
            transform     = valid_transforms, 
    )

    sampler = None

    trainloader = DataLoader(
            trainset, 
            batch_size     = CFG['batch_size_t'], 
            shuffle        = False, 
            num_workers    = CFG['num_workers'], 
            worker_init_fn = seed_worker, 
            pin_memory     = True,
            sampler        = sampler, 
            drop_last      = True
    )

    validloader = DataLoader(
            validset, 
            batch_size     = CFG['batch_size_v'], 
            shuffle        = False, 
            num_workers    = CFG['num_workers'], 
            worker_init_fn = seed_worker, 
            pin_memory     = True,
            drop_last      = False
    )
    
    CFG['no_batches'] = len(trainloader)

    model = RadarSignalsModel(
       model_name      = CFG['model_name'],
       n_targets       = CFG['n_targets'],
       pretrained      = True,
    )

    model.to(DEVICE)

    optimizer = get_optimizer(model.parameters(), CFG)
    scheduler = get_scheduler(optimizer, CFG)
    criterion = get_criterion(CFG)

    if CFG['use_swa']:
        swa_best_model = None
        swa_best_score = -np.inf
        swa_best_predictions = None
        swa_model = AveragedModel(model)

    if CFG['use_apex']:
        scaler = GradScaler()
    else:
        scaler = None

    best_model = None
    best_score  = -np.inf
    best_predictions = None
    train_losses_plot, valid_losses_plot = [], []
    train_scores_plot, valid_scores_plot = [], []
    for epoch in range(CFG['epochs']):
        start_epoch = time.time()

        if CFG['update_per_batch'] == False: scheduler.step(epoch)

        train_avg_accuracy, train_losses, train_scores = train_epoch(model, trainloader, optimizer, criterion, scheduler, epoch, DEVICE, scaler, CFG, logger)
        valid_avg_accuracy, valid_predictions, valid_losses, valid_scores = valid_epoch(model, validloader, criterion, DEVICE, CFG, logger)

        train_losses_plot.append(train_losses)
        train_scores_plot.append(train_scores)
        valid_losses_plot.append(valid_losses)
        valid_scores_plot.append(valid_scores)

        accuracy  = accuracy_score(valid_labels, valid_predictions) 
        precision = precision_score(valid_labels, valid_predictions, average = 'weighted')
        recall    = recall_score(valid_labels, valid_predictions, average = 'weighted')

        elapsed = time.time() - start_epoch

        logger.print("Valid Labels Distribution: {}, {}".format(*np.unique(valid_labels, return_counts = True)))
        logger.print("Valid Predictions Distribution: {}, {}".format(*np.unique(valid_predictions, return_counts = True)))
        logger.print(f'Epoch {epoch + 1} - Train Average Loss: {train_avg_accuracy:.3f}, Valid Average Loss: {valid_avg_accuracy:.3f}, Epoch Time: {elapsed:.3f}s')
        logger.print(f'Epoch {epoch + 1} - Accuracy: {accuracy:.3f}, Precision: {precision:.3f}, Recall: {recall:.3f}')

        if accuracy > best_score:
            logger.print(f"Saved Best Model: {accuracy:.2f}")
            best_score = accuracy
            best_predictions = valid_predictions
            oof[valid_idx] = best_predictions

            best_model = {
                'model':     {key: value.cpu() for key, value in model.state_dict().items()},
                'oof_proba':  best_predictions,
                'oof_labels': valid_labels,
                'oof_ids':    valid_ids
            }

        if CFG['use_swa'] and epoch + 1 in CFG['swa_epoch']:
            swa_model.update_parameters(model)
            valid_swa_accuracy, valid_swa_predictions, _, _ = valid_epoch(swa_model, validloader, criterion, DEVICE, CFG, logger)

            swa_accuracy  = accuracy_score(valid_labels, valid_swa_predictions) 
            swa_precision = precision_score(valid_labels, valid_swa_predictions, average = 'weighted')
            swa_recall    = recall_score(valid_labels, valid_swa_predictions, average = 'weighted')

            logger.print(f'Epoch {epoch + 1} - Baseline Accuracy: {accuracy:.3f} - SWA Accuracy: {swa_accuracy:.3f}, SWA Precision: {swa_precision:.3f}, SWA Recall: {swa_recall:.3f}')
            if swa_accuracy > swa_best_score:
                logger.print(f'Saved Best Model: {swa_accuracy:.2f}')
                swa_best_model = swa_model
                swa_best_score = swa_accuracy

        if train_avg_accuracy - valid_avg_accuracy > 30: 
            logger.print("[EXIT] Overfitting Condition...")
            break

    if CFG['use_swa']:
        torch.optim.swa_utils.update_bn(trainloader, swa_best_model, device = DEVICE)
        swa_best_score, best_swa_predictions, _, _ = valid_epoch(swa_best_model, validloader, criterion, DEVICE, CFG, logger)
        swa_oof[valid_idx] = best_swa_predictions

        best_swa_model = {
            'swa_model':  {key: value.cpu() for key, value in swa_best_model.state_dict().items()},
            'oof_proba':  best_swa_predictions,
            'oof_labels': valid_labels,
            'oof_ids':    valid_ids
        }

        swa_accuracy  = accuracy_score(valid_labels, best_swa_predictions) 
        swa_precision = precision_score(valid_labels, best_swa_predictions, average = 'weighted')
        swa_recall    = recall_score(valid_labels, best_swa_predictions, average = 'weighted')
        logger.print(f'SWA Accuracy: {swa_accuracy:.3f}, SWA Precision: {swa_precision:.3f}, SWA Recall: {swa_recall:.3f}')

    if CFG['save_to_log']:
        xcoords = [x for x in range(1, epoch + 1)]
        plt.clf()
        plt.plot(train_losses_plot, '-D', markevery = xcoords, label = 'Train Losses')
        plt.plot(valid_losses_plot, '-D', markevery = xcoords, label = 'Valid Losses')
        plt.title(f"[Model {CFG['id']}, Fold {fold}]: Losses Plot")
        plt.xlabel("Epochs")
        plt.ylabel("Losses")
        plt.legend(loc = True)
        plt.savefig(os.path.join(
            PATH_TO_MODELS,
            f"losses_plot_model_{CFG['id']}_fold_{fold}.png"
        ))
        plt.clf()
        plt.plot(train_scores_plot, '-D', markevery = xcoords, label = 'Train Scores')
        plt.plot(valid_scores_plot, '-D', markevery = xcoords, label = 'Valid Scores')
        plt.title(f"[Model {CFG['id']}, Fold {fold}]: Scores Plot")
        plt.xlabel("Epochs")
        plt.ylabel("Scores")
        plt.legend(loc = True)
        plt.savefig(os.path.join(
            PATH_TO_MODELS,
            f"scores_plot_model_{CFG['id']}_fold_{fold}.png"
        ))

    if CFG['use_swa']:
        return oof, best_score, best_model, swa_oof, swa_accuracy, best_swa_model
    else:
        return oof, best_score, best_model, None, None, None

def run(GPU, CFG, GLOBAL_LOGGER, PATH_TO_MODELS, logger):
    seed_everything(SEED)

    DEVICE = torch.device('cuda:{}'.format(GPU) if torch.cuda.is_available() else 'cpu')
    train  = pd.read_csv(PATH_TO_TRAIN_META)
    train["path"]  = train["id"].apply(lambda x: os.path.join(PATH_TO_TRAIN_IMAGES, x))
    train["label"] = train["label"].apply(lambda x: x - 1)
    # train = train[train['all_wrong'] == 1].reset_index(drop = True)
    # train = train.sample(100, random_state = SEED).reset_index(drop = True)

    PATH_TO_OOF = f"logs/stage-{STAGE}/gpu-{GPU}/oof.csv"
    logger.print(f"[GPU {GPU}]: Config File")
    logger.print(CFG)
    
    train = generate_folds(
        data         = train, 
        skf_column   = 'label', 
        n_folds      = CFG['n_folds'], 
        random_state = SEED
    )

    oof     = np.zeros((train.shape[0],), dtype = np.float32)
    swa_oof = copy.deepcopy(oof)

    best_models, best_swa_models = [], []
    fold_accuracies, swa_fold_accuracies = [], []
    for fold in range(CFG['n_folds']):
        oof, fold_accuracy, best_model, swa_oof, swa_accuracy, swa_best_model = train_fold(CFG, train, fold, oof, logger, PATH_TO_MODELS, DEVICE, swa_oof)
        
        best_models.append((fold_accuracy, copy.deepcopy(best_model)))
        fold_accuracies.append(fold_accuracy)

        if CFG['use_swa']:
            best_swa_models.append((swa_accuracy, copy.deepcopy(swa_best_model)))
            swa_fold_accuracies.append(swa_accuracy)

        if CFG['one_fold']: break

    if CFG['one_fold'] == False:
        predictions = pd.read_csv(PATH_TO_OOF)
        predictions['model_{}'.format(CFG['id'])] = oof + 1

        if CFG['use_swa']: 
             predictions['swa_model_{}'.format(CFG['id'])] = swa_oof + 1

        predictions.to_csv(PATH_TO_OOF, index = False)

    OUTPUT["oof-accuracy"]  = accuracy_score(train['label'].values, oof)
    OUTPUT["oof-precision"] = precision_score(train['label'].values, oof, average = 'weighted')
    OUTPUT["oof-recall"]    = recall_score(train['label'].values, oof, average = 'weighted')

    OUTPUT["cross-validation"] = fold_accuracies

    if CFG['use_swa']:
         OUTPUT["cross-validation"] = fold_accuracies + swa_fold_accuracies

    GLOBAL_LOGGER.append(CFG, OUTPUT)

    if CFG['use_swa'] == False:
        return RD(np.mean(fold_accuracies)), best_models 
    else:
        return RD(np.mean(fold_accuracies)), best_models, RD(np.mean(swa_fold_accuracies)), best_swa_models

if __name__ == "__main__":
    QUIET = True
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
        'optimizer': "AdamW",
        'scheduler': "CosineAnnealingWarmRestarts",
        
        'LR': 7e-05,
        'T_0': 74,
        'T_max': 10,
        'T_mult': 2,
        'min_lr': 1e-6,
        'max_lr': 1e-4,
        'no_batches': 'NA',
        'warmup_epochs': 1,
        'cosine_epochs': 11,
        'epochs' : 12,
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
        'swa_epoch': [3, 5, 6, 10, 11, 12],

        # Adaptive Sharpness-Aware Minimization
        'use_sam':  False,
        'use_asam': True,
        'asam_rho': 0.1,

        # Parameters for script control
        'print_freq': 50,
        'one_fold': False,
        'use_apex': True,
        'distributed_training': DISTRIBUTED_TRAINING, # python -m torch.distributed.launch --nproc_per_node=1 train.py
        'save_to_log': SAVE_TO_LOG
    }

    if SAVE_TO_LOG:
        PATH_TO_MODELS = os.path.join(PATH_TO_MODELS, 'gpu-{}/model_{}'.format(GPU, CFG['id']))
        if os.path.isdir(PATH_TO_MODELS) == False: os.makedirs(PATH_TO_MODELS)
        logger = Logger(os.path.join(PATH_TO_MODELS, 'model_{}.log'.format(CFG['id'])), distributed = QUIET)
    else:
        logger = Logger(distributed = QUIET)

    if CFG['use_swa']:
        accuracy, best_models, swa_accuracy, best_swa_models = run(GPU, CFG, GLOBAL_LOGGER, PATH_TO_MODELS, logger)
        print(f"Accuracy: {accuracy}")
        print(f"SWA Accuracy: {swa_accuracy}")  

        if CFG['save_to_log']:
            for fold, (accuracy, model) in enumerate(best_models): 
                torch.save(
                    model, 
                    os.path.join(PATH_TO_MODELS, f"model_{CFG['id']}_name_{CFG['model_name']}_fold_{fold}_accuracy_{accuracy:.2f}.pth")
                )
            
            for fold, (accuracy, model) in enumerate(best_swa_models): 
                torch.save(
                    model, 
                    os.path.join(PATH_TO_MODELS, f"swa_model_{CFG['id']}_name_{CFG['model_name']}_fold_{fold}_accuracy_{accuracy:.2f}.pth")
                )


    else:
        accuracy, best_models = run(GPU, CFG, GLOBAL_LOGGER, PATH_TO_MODELS, logger)
        print(f"Accuracy: {accuracy}")

        if CFG['save_to_log']:
            for fold, (accuracy, model) in enumerate(best_models): 
                torch.save(
                    model, 
                    os.path.join(PATH_TO_MODELS, f"model_{CFG['id']}_name_{CFG['model_name']}_fold_{fold}_accuracy_{accuracy:.2f}.pth")
                )
                
    
