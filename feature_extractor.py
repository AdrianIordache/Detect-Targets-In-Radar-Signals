from utils import *
from models  import *
from dataset import *

CFG = {
    'model_name': 'swin_large_patch4_window12_384_in22k',
    'size': 384,
    'batch_size_t': 3,
    'batch_size_v': 56,

    'n_tragets': 5,
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

def extract_embeddings(dataset: pd.DataFrame, embedding_size: int = 1536, typs: str = 'train'):
    STAGE   = 1
    GPU     = 1 
    VERSION = 42
    FOLDS   = [(0, "0.77"), (1, "0.76"), (2, "0.76"), (3, "0.76"), (4, "0.76")]

    tic = time.time()
    for fold, accuracy in FOLDS:
        embeddings = np.zeros((dataset.shape[0], embedding_size))
        PATH_TO_MODEL = f"models/stage-{STAGE}/gpu-{GPU}/model_{VERSION}/model_{VERSION}_name_{CFG['model_name']}_fold_{fold}_accuracy_{accuracy}.pth"
        print("Current Model Inference: ", PATH_TO_MODEL)

        states = torch.load(PATH_TO_MODEL, map_location = torch.device('cpu'))

        model = RadarSignalsModel(
           model_name      = CFG['model_name'],
           n_targets       = CFG['n_tragets'],
           pretrained      = False,
        ).to(DEVICE)

        model.load_state_dict(states['model'])       
        model.eval() 

        test_transforms = A.Compose([
            A.Resize(CFG['size'], CFG['size']),
            A.Normalize(mean = [0.485, 0.456, 0.406], std  = [0.229, 0.224, 0.225]),
            ToTensorV2()
        ])

        testset = RadarSignalsDataset(
                dataset, 
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
                preds = model(images, embeddings = True).squeeze(1)
                
            preds = preds.cpu().numpy()
            predictions.extend(preds)

            end = time.time()
            if batch % CFG['print_freq'] == 0 or batch == (len(testloader) - 1):
                print('[GPU {0}][INFERENCE][{1}/{2}], Elapsed {remain:s}'
                      .format(DEVICE, batch, len(testloader), remain = timeSince(start, float(batch + 1) / len(testloader))))

        images = images.detach().cpu()
        embeddings[:, :] = predictions

        embeddings_csv = pd.DataFrame(embeddings, columns = ['X_{}'.format(x) for x in range(embedding_size)])
        embeddings_csv["id"]   = dataset['id'].values

        embeddings_csv.to_csv(
            os.path.join(PATH_TO_EMBEDDINGS, f'{typs}_stage_{STAGE}_gpu_{GPU}_version_{VERSION}_fold_{fold}_baseline_{accuracy}.csv'),
                index = False
        )

def extract_noisy_feature(dataset: pd.DataFrame):
    oof = np.zeros((dataset.shape[0], CFG['n_folds']))

    GPU     = 0 
    VERSION = 0
    FOLDS   = [(0, "0.97"), (1, "0.96"), (2, "0.97"), (3, "0.97"), (4, "0.96")]

    tic = time.time()
    tta_transforms = [[]]
    for fold, accuracy in FOLDS:
        PATH_TO_MODEL = f"data/detect-targets-in-radar-signals/denoising/models/gpu-{GPU}/model_{VERSION}/model_{VERSION}_name_{CFG['model_name']}_fold_{fold}_accuracy_{accuracy}.pth"
        print("Current Model Inference: ", PATH_TO_MODEL)

        states = torch.load(PATH_TO_MODEL, map_location = torch.device('cpu'))

        model = RadarSignalsModel(
           model_name      = CFG['model_name'],
           n_targets       = CFG['n_tragets'],
           pretrained      = False,
        ).to(DEVICE)
        
        model.load_state_dict(states['model'])       
        model.eval() 

        tta_predictions = np.zeros((dataset.shape[0], len(tta_transforms)))
        for tta_index, tta in enumerate(tta_transforms):
            print(f"Current TTA {tta_index + 1}/{len(tta_transforms)}")
            
            test_transforms = A.Compose(
                tta + [
                A.Resize(CFG['size'], CFG['size']),
                A.Normalize(mean = [0.485, 0.456, 0.406], std  = [0.229, 0.224, 0.225]),
                ToTensorV2()
            ])


            testset = RadarSignalsDataset(
                    dataset, 
                    train     = False,
                    transform = test_transforms, 
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
                
                preds = torch.round(torch.sigmoid(preds))
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

    dataset['isNoisy'] = final_predictions
    dataset['isNoisy'] = dataset['isNoisy'].astype(int)
    return dataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', 
        dest = 'gpu', nargs = '+', type = int, 
        default = 0, help = "GPU enable for running the procces"
    )

    args  = parser.parse_args()
    RANK  = args.gpu[0]
   
    DEVICE = torch.device('cuda:{}'.format(RANK) 
        if torch.cuda.is_available() else 'cpu'
    )

    if 0:
        train = pd.read_csv(PATH_TO_TRAIN_META)
        train["path"]  = train["id"].apply(lambda x: os.path.join(PATH_TO_TRAIN_IMAGES, x))
        train = extract_noisy_feature(train)
        train.to_csv(PATH_TO_TRAIN_FE, index = False)

        test  = pd.read_csv(PATH_TO_TEST_META)
        test["path"] = test["id"].apply(lambda x: os.path.join(PATH_TO_TEST_IMAGES, x))
        test  = extract_noisy_feature(test)
        test.to_csv(PATH_TO_TEST_FE, index = False)

    if 1:
        train = pd.read_csv(PATH_TO_TRAIN_META)
        train["path"]  = train["id"].apply(lambda x: os.path.join(PATH_TO_TRAIN_IMAGES, x))
        extract_embeddings(train, typs = 'train')

        test = pd.read_csv(PATH_TO_TEST_META)
        test["path"]  = test["id"].apply(lambda x: os.path.join(PATH_TO_TEST_IMAGES, x))
        extract_embeddings(test, typs = 'test')
