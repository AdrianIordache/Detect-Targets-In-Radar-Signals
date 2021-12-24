from utils   import *
from models  import *
from dataset import *

def inference(MODELS, CFG, RANK, GPU, VERBOSE = False):
    test = pd.read_csv(PATH_TO_TEST_META)
    test['path'] = test['id'].apply(lambda x: os.path.join(PATH_TO_TEST_IMAGES, x))
    
    if len(CFG['train_transforms']) != 0:
        tta_transforms = [
            [],
            [A.HorizontalFlip(p = 1)], 
            [A.VerticalFlip(p = 1)],
            [A.RandomRotate90(p = 0.5)]
        ]
    else: 
        tta_transforms = [[]]

    DEVICE = torch.device('cuda:{}'.format(RANK) 
        if torch.cuda.is_available() else 'cpu'
    )

    VERSION = CFG['id']
    oof = np.zeros((test.shape[0], len(tta_transforms), len(MODELS)))

    tic = time.time()
    for i_fold, (accuracy, states) in enumerate(MODELS):
        if VERBOSE: print("Current Model Inference: Fold", i_fold)

        model = RadarSignalsModel(
           model_name      = CFG['model_name'],
           n_targets       = CFG['n_targets'],
           pretrained      = False,
        ).to(DEVICE)
        
        if CFG['use_swa']:
            model = AveragedModel(model)

        model.load_state_dict(states['swa_model'])       
        model.eval() 

        tta_predictions = np.zeros((test.shape[0], len(tta_transforms)))
        for tta_index, tta in enumerate(tta_transforms):
            if VERBOSE: print(f"Current TTA {tta_index + 1}/{len(tta_transforms)}")
            
            test_transforms = A.Compose(
                tta + [
                A.Resize(CFG['size'], CFG['size']),
                A.Normalize(mean = MEANS_IMAGENET, std  = STDS_IMAGENET),
                ToTensorV2()
            ])

            testset = RadarSignalsDataset(
                    test, 
                    train         = False, 
                    transform     = test_transforms, 
            )

            testloader = DataLoader(
                    testset, 
                    batch_size     = CFG['batch_size_v'], 
                    shuffle        = False, 
                    num_workers    = CFG['num_workers'], 
                    worker_init_fn = seed_worker, 
                    pin_memory     = True,
                    drop_last      = False
            )
            
            predictions = []
            start = end = time.time()
            for batch, (images) in enumerate(testloader):
                images = images.to(DEVICE)

                with torch.no_grad():
                    preds = model(images).squeeze(1)
                
                preds = torch.argmax(preds, dim = 1).cpu().numpy() + 1
                predictions.extend(preds)

                end = time.time()
                if VERBOSE:
                    if batch % CFG['print_freq'] == 0 or batch == (len(testloader) - 1):
                        print('[GPU {0}][INFERENCE][{1}/{2}], Elapsed {remain:s}'
                              .format(DEVICE, batch, len(testloader), 
                                remain = timeSince(start, float(batch + 1) / len(testloader)))
                        )
            
            tta_predictions[:, tta_index] = predictions    
            del testset, testloader, predictions, preds
            gc.collect()
        
        oof[:, :, i_fold] = tta_predictions
        free_gpu_memory(DEVICE)
        
        del states, tta_predictions
        gc.collect()
    
    oof  = oof.reshape((test.shape[0], -1))
    path = f'models/stage-{STAGE}/gpu-{GPU}/model_{VERSION}/'

    votes = pd.DataFrame(oof.astype(int), columns = [f"vote_{vote}" for vote in range(oof.shape[1])])
    votes['id'] = copy.deepcopy(test['id'])
    votes_name = f'votes_stage_{STAGE}_gpu_{GPU}_version_{VERSION}.csv'
    if CFG['use_swa']: votes_name = 'swa_' + votes_name
    votes.to_csv(os.path.join(path, votes_name), index = False)

    final_predictions = []
    for i in range(oof.shape[0]):
        values, counts = np.unique(oof[i], return_counts = True)
        index = np.argmax(counts)
        final_predictions.append(values[index])

    submission = pd.DataFrame(columns = ['id', 'label'])
    submission['id']    = copy.deepcopy(test["id"])
    submission['label'] = final_predictions
    submission['label'] = submission['label'].astype(int)
    if VERBOSE: display(submission)
    toc = time.time()

    submission_name = f'submission_stage_{STAGE}_gpu_{GPU}_version_{VERSION}.csv'
    if CFG['use_swa']: submission_name = 'swa_' + submission_name
    submission.to_csv(os.path.join(path, submission_name), index = False)

    if VERBOSE: print("Inference Time: {}'s".format(toc - tic))


if __name__ == "__main__":
    STAGE   = 3
    GPU     = 1
    VERSION = 1
    FOLDS   = [(0, '0.77'), (1, '0.77'), (2, '0.77'), (3, '0.76'), (4, '0.77'), 
               (5, '0.77'), (6, '0.77'), (7, '0.76'), (8, '0.77'), (9, '0.76')]

    CFG = {
        'id': VERSION,
        'model_name': 'swin_large_patch4_window12_384_in22k',
        'size': 384,
        'batch_size_t': 3,
        'batch_size_v': 52,

        'n_targets': 5,
        'num_workers': 4,
        'n_folds': 5,

        # Augumentations and other obserrvations for experiment
        'train_transforms': [],
        'valid_transforms': [],
        'observations':   None,

        # Stochastic Weight Averaging
        'use_swa': True,

        # Parameters for script control
        'print_freq': 50,
        'one_fold': False,
        'use_apex': False,
    }

    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', 
        dest = 'gpu', nargs = '+', type = int, 
        default = 0, help = "GPU enable for running the procces"
    )

    args  = parser.parse_args()
    RANK  = args.gpu[0]
    
    MODELS = []
    for fold_idx, (fold, accuracy) in enumerate(FOLDS):
        PATH_TO_MODEL = f"models/stage-{STAGE}/gpu-{GPU}/model_{VERSION}/swa_model_{VERSION}_name_{CFG['model_name']}_fold_{fold}_accuracy_{accuracy}.pth"
        states = torch.load(PATH_TO_MODEL, map_location = torch.device('cpu'))
        MODELS.append((accuracy, copy.deepcopy(states)))

    inference(MODELS, CFG, RANK, GPU, VERBOSE = True)
