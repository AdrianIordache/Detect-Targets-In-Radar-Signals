from utils   import *
from models  import *
from dataset import *

CFG = {
    'model_name': "tf_efficientnet_b0_ns", # 'swin_large_patch4_window12_384_in22k',
    'size': 384,
    'batch_size_t': 3,
    'batch_size_v': 16,

    'n_tragets': 5,
    'num_workers': 4,
    'n_folds': 3,

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

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', 
        dest = 'gpu', nargs = '+', type = int, 
        default = 0, help = "GPU enable for running the procces"
    )

    args = parser.parse_args()
    GPU  = args.gpu[0]
   
    DEVICE = torch.device('cuda:{}'.format(GPU) 
        if torch.cuda.is_available() else 'cpu'
    )

    test = pd.read_csv(PATH_TO_TEST_META)
    test['path'] = test['id'].apply(lambda x: os.path.join(PATH_TO_TEST_IMAGES, x))
    tta_transforms = [[]]

    STAGE   = 0
    GPU     = 0 
    VERSION = 0
    FOLDS   = [(0, "0.26"), (1, "0.30"), (2, "0.22")]

    oof = np.zeros((test.shape[0], CFG['n_folds']))

    tic = time.time()
    for fold, loss in FOLDS:
        PATH_TO_MODEL = f"models/stage-{STAGE}/gpu-{GPU}/model_{VERSION}/model_{VERSION}_name_{CFG['model_name']}_fold_{fold}_accuracy_{loss}.pth"
        print("Current Model Inference: ", PATH_TO_MODEL)

        states = torch.load(PATH_TO_MODEL, map_location = torch.device('cpu'))

        model = RadarSignalsModel(
           model_name      = CFG['model_name'],
           n_targets       = CFG['n_tragets'],
           pretrained      = False,
        ).to(DEVICE)
        
        model.load_state_dict(states['model'])       
        model.eval() 

        tta_predictions = np.zeros((test.shape[0], len(tta_transforms)))
        for tta_index, tta in enumerate(tta_transforms):
            print(f"Current TTA {tta_index + 1}/{len(tta_transforms)}")
            
            test_transforms = A.Compose(
                tta + [
                A.Resize(CFG['size'], CFG['size']),
                A.Normalize(mean = [0.485, 0.456, 0.406], std  = [0.229, 0.224, 0.225]),
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
                if batch % CFG['print_freq'] == 0 or batch == (len(testloader) - 1):
                    print('[GPU {0}][INFERENCE][{1}/{2}], Elapsed {remain:s}'
                          .format(DEVICE, batch, len(testloader), remain = timeSince(start, float(batch + 1) / len(testloader))))
            
            tta_predictions[:, tta_index] = predictions    
            del testset, testloader, predictions, preds
            gc.collect()
        
        oof[:, fold] = np.mean(tta_predictions, axis = 1)
        free_gpu_memory(DEVICE)
        
        del states, tta_predictions
        gc.collect()
        
    final_predictions = []
    for i in range(oof.shape[0]):
        values, counts = np.unique(oof[i], return_counts = True)
        index = np.argmax(counts)
        final_predictions.append(values[index])

    submission = pd.DataFrame(columns = ['id', 'label'])
    submission['id']    = copy.deepcopy(test["id"])
    submission['label'] = final_predictions
    display(submission)
    toc = time.time()

    submission.to_csv(f'models/stage-{STAGE}/gpu-{GPU}/model_{VERSION}/submission_{VERSION}.csv', index = False)
    print("Inference Time: {}'s".format(toc - tic))