import sys
sys.path.append("..")

from utils import *
from fastai.vision.all import *

def seed_everything(SEED = 42, reproducible = True, dls = None):
    set_seed(SEED, reproducible)
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms = True
    
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)
    if dls is not None: dls.rng.seed(SEED)

def get_data(data, FOLD, CFG):
    data_copy = data.copy()
    data_copy['is_valid'] = (data_copy['fold'] == FOLD)

    dls = ImageDataLoaders.from_df(
        df          = data_copy,             # pass in train DataFrame
        valid_col   = 'is_valid',            # add is_valid for validation fold
        seed        = SEED, 
        fn_col      = 'path',                # filename/path is in the second column of the DataFrame
        label_col   = 'label',               # label is in the first column of the DataFrame
        y_block     = CategoryBlock,         # The type of target
        bs          = CFG['batch_size_t'],   # pass in batch size
        num_workers = 8,
        item_tfms   = Resize(CFG['size']),   # pass in item_tfms
        batch_tfms  = setup_aug_tfms([Brightness(), Contrast(), Hue(), Saturation()])
    )
    
    seed_everything(SEED = SEED, reproducible = True, dls = dls)
    return dls

def get_learner(data, FOLD, CFG):
    data  = get_data(data, FOLD, CFG)
    model = timm.create_model(CFG['model_name'], pretrained = True, num_classes = data.c)
    learn = Learner(data, model, loss_func = CrossEntropyLossFlat(), metrics = accuracy).to_fp16()
    return learn

def inference(learn, data, test, CFG):
    dls = ImageDataLoaders.from_df(
        df          = data,                  # pass in train DataFrame
        seed        = SEED, 
        fn_col      = 'path',                # filename/path is in the second column of the DataFrame
        label_col   = 'label',               # label is in the first column of the DataFrame
        y_block     = CategoryBlock,         # The type of target
        bs          = CFG['batch_size_v'],   # pass in batch size
        num_workers = 8,
        item_tfms   = Resize(CFG['size']),   # pass in item_tfms
        batch_tfms  = setup_aug_tfms([Brightness(), Contrast(), Hue(), Saturation()])
    )
    
    test_dl = dls.test_dl(test)

    preds, _ = learn.tta(dl = test_dl, n = CFG['n_tta'], use_max = True)

    return np.argmax(preds.float().numpy(), axis = 1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', 
        dest = 'gpu', nargs = '+', type = int, 
        default = 0, help = "GPU enable for running the procces"
    )

    args = parser.parse_args()
    GPU  = args.gpu[0]
    torch.cuda.set_device(GPU)
    
    CFG = {
        'id': 0,
        'model_name': 'efficientnet_b0',
        'size': 384,
        'batch_size_t': 3,
        'batch_size_v': 3,

        'epochs':  20,
        'n_folds': 5,
        'n_tta':   5,
        'debug':   True
    }

    seed_everything(SEED = SEED, reproducible = True)

    path = Path('../data/detect-targets-in-radar-signals/')
    train_df = pd.read_csv(os.path.join(path, 'train.csv'))
    test_df  = pd.read_csv(os.path.join(path, 'test.csv'))

    train_df['path'] = train_df['id'].map(lambda x: os.path.join(path, 'train', x))
    test_df['path'] = test_df['id'].map(lambda x: os.path.join(path, 'test', x))

    if CFG['debug']:
        CFG['epochs'] = 1
        train_df = train_df.sample(50, random_state = SEED).reset_index(drop = True)
        test_df  = test_df.sample(50, random_state = SEED).reset_index(drop = True)

    train_df = generate_folds(train_df, 'label', n_folds = CFG['n_folds'])
    display(train_df)

    oof_probas, oof_labels = [], [] 
    votes = np.zeros((len(test_df), CFG['n_folds']))
    for fold in range(CFG['n_folds']):
        minimum_lr, steep_lr, valley_lr, slide_lr = get_learner(train_df, FOLD = fold, CFG = CFG).lr_find(end_lr = 5e-2, suggest_funcs = (minimum, steep, valley, slide), num_it = 200)

        print(f'Fold {fold} results')
        
        learn = get_learner(train_df, FOLD = fold, CFG = CFG)

        learn.fit_one_cycle(CFG['epochs'], valley_lr, 
            cbs = [SaveModelCallback(), EarlyStoppingCallback(monitor = "accuracy",  patience = 5)]
       ) 
        
        probas, targets = learn.get_preds()

        probas   = np.argmax(probas.detach().cpu().numpy(), axis = 1)
        targets  = targets.detach().cpu().numpy()
        valid_accuracy = accuracy_score(targets, probas) 
        print(f"Fold: {fold}, Accuracy: {valid_accuracy}")

        oof_probas.extend(probas)
        oof_labels.extend(targets)
        
        learn = learn.to_fp32()
        # learn.save(f"saved_model_{CFG['id']}_name_{CFG['model_name']}_fold_{fold}_accuracy_{valid_accuracy:.2f}")
        learn.export(f"models/exported_model_{CFG['id']}_name_{CFG['model_name']}_fold_{fold}_accuracy_{valid_accuracy:.2f}")
        
        oof_votes = inference(learn, train_df, test_df, CFG)
        votes[:, fold] = oof_votes

        del learn
        torch.cuda.empty_cache()
        gc.collect()

final_predictions = []
for i in range(votes.shape[0]):
    values, counts = np.unique(votes[i], return_counts = True)
    index = np.argmax(counts)
    final_predictions.append(values[index])

submission = pd.DataFrame(columns = ['id', 'label'])
submission['id']    = test_df['id']
submission['label'] = final_predictions
submission['label'] = submission['label'].astype(int)
submission.to_csv(f"submissions/fastai_{CFG['model_name']}_submission_{CFG['id']}.csv", index = False)

results = {
    'probas': oof_probas,
    'labels': oof_labels
}

pd.DataFrame(results).to_csv('external_results.csv', index = False)