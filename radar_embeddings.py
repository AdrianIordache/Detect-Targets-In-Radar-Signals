#!/usr/bin/env python
# coding: utf-8

# In[2]:


import sys

get_ipython().system('cp ../input/rapids/rapids.0.17.0 /opt/conda/envs/rapids.tar.gz')
get_ipython().system('cd /opt/conda/envs/ && tar -xzvf rapids.tar.gz > /dev/null')
get_ipython().system('rm /opt/conda/envs/rapids.tar.gz')

sys.path += ["/opt/conda/envs/rapids/lib/python3.7/site-packages"]
sys.path += ["/opt/conda/envs/rapids/lib/python3.7"]
sys.path += ["/opt/conda/envs/rapids/lib"]
get_ipython().system('cp /opt/conda/envs/rapids/lib/libxgboost.so /opt/conda/lib/')


# In[3]:


from cuml.svm import SVC


# In[4]:


import gc
import os
import random
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import  accuracy_score
from IPython.display import display

import warnings
warnings.filterwarnings("ignore")


# In[5]:


PATH_TO_DATA       = '../input/detect-targets-in-radar-signals/'
PATH_TO_EMBEDDINGS = '../input/radardataset/'

PATH_TO_TRAIN_META = os.path.join(PATH_TO_DATA, "train.csv") 
PATH_TO_TEST_META  = os.path.join(PATH_TO_DATA, "test.csv") 

PATH_TO_FOLDS      = os.path.join(PATH_TO_EMBEDDINGS, "10_folds.csv") 

def seed_everything(SEED = 42):
    random.seed(SEED)
    np.random.seed(SEED)
    os.environ['PYTHONHASHSEED'] = str(SEED)

SEED = 42
seed_everything(SEED)


# In[7]:


# c in [0.001, 0.002, 0.004, 0.006, 0.008, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.2, 0.4, 0.6, 0.8, 1, 2, 3, 4, 5, 10, 15]
# [(0.7761290322580645, 0.02), (0.7806451612903226, 0.06), (0.7825806451612903, 0.04), (0.7812903225806451, 0.06), (0.7741935483870968, 0.7), (0.7858064516129032, 0.06), (0.7761290322580645, 0.2), (0.7709677419354839, 0.1), (0.7793548387096774, 0.9), (0.7703225806451612, 0.09)]
FOLD_MODELS = [
    [(3, 0, 1, False,  0, "0.766"), (3, 0, 1, True,  0, "0.767"), (3, 1, 1, False,  0, "0.767"), (3, 1, 1, True,  0, "0.774")], # C = 0.02
    [(3, 0, 1, False,  1, "0.768"), (3, 0, 1, True,  1, "0.778"), (3, 1, 1, False,  1, "0.773"), (3, 1, 1, True,  1, "0.774")], # C = 0.06
    [(3, 0, 1, False,  2, "0.770"), (3, 0, 1, True,  2, "0.772"), (3, 1, 1, False,  2, "0.774"), (3, 1, 1, True,  2, "0.778")], # C = 0.04
    [(3, 0, 1, False,  3, "0.761"), (3, 0, 1, True,  3, "0.771"), (3, 1, 1, False,  3, "0.771"), (3, 1, 1, True,  3, "0.763")], # C = 0.06
    [(3, 0, 1, False,  4, "0.753"), (3, 0, 1, True,  4, "0.754"), (3, 1, 1, False,  4, "0.759"), (3, 1, 1, True,  4, "0.772")], # C = 0.7
    [(3, 0, 1, False,  5, "0.770"), (3, 0, 1, True,  5, "0.775"), (3, 1, 1, False,  5, "0.779"), (3, 1, 1, True,  5, "0.783")], # C = 0.06
    [(3, 0, 1, False,  6, "0.768"), (3, 0, 1, True,  6, "0.772"), (3, 1, 1, False,  6, "0.766"), (3, 1, 1, True,  6, "0.778")], # C = 0.2
    [(3, 0, 1, False,  7, "0.767"), (3, 0, 1, True,  7, "0.755"), (3, 1, 1, False,  7, "0.767"), (3, 1, 1, True,  7, "0.759")], # C = 0.1
    [(3, 0, 1, False,  8, "0.768"), (3, 0, 1, True,  8, "0.773"), (3, 1, 1, False,  8, "0.770"), (3, 1, 1, True,  8, "0.775")], # C = 0.9
    [(3, 0, 1, False,  9, "0.761"), (3, 0, 1, True,  9, "0.768"), (3, 1, 1, False,  9, "0.761"), (3, 1, 1, True,  9, "0.761")], # C = 0.09
]

Cs = [0.02, 0.06, 0.04, 0.06, 0.7, 0.06, 0.2, 0.1, 0.9, 0.09]
votes = np.zeros((5500, len(FOLD_MODELS)))

mean_accuracy = []
best_thresholds = []
for MODELS, C in zip(FOLD_MODELS, Cs):
    trainset = pd.read_csv(PATH_TO_TRAIN_META)
    testset  = pd.read_csv(PATH_TO_TEST_META)

    folds    = pd.read_csv(PATH_TO_FOLDS)
    trainset['fold'] = folds['fold']
    del folds; gc.collect()

    for i, (stage, gpu, version, is_swa, fold, baseline) in enumerate(MODELS):
        print("Stage {}, GPU {}, Model {}, Fold {}, Baseline {}".format(stage, gpu, version, fold, baseline))
        if is_swa:
            train_features  = f'swa_train_stage_{stage}_gpu_{gpu}_version_{version}_fold_{fold}_baseline_{baseline}.csv'
            test_features   = f'swa_test_stage_{stage}_gpu_{gpu}_version_{version}_fold_{fold}_baseline_{baseline}.csv'
        else:
            train_features  = f'train_stage_{stage}_gpu_{gpu}_version_{version}_fold_{fold}_baseline_{baseline}.csv'
            test_features   = f'test_stage_{stage}_gpu_{gpu}_version_{version}_fold_{fold}_baseline_{baseline}.csv'

        train_embeddings_path = os.path.join(PATH_TO_EMBEDDINGS, train_features)
        test_embeddings_path  = os.path.join(PATH_TO_EMBEDDINGS, test_features)

        train_embeddings = pd.read_csv(train_embeddings_path)
        trainset         = pd.merge(trainset, train_embeddings, on = 'id')

        test_embeddings  = pd.read_csv(test_embeddings_path)
        testset          = pd.merge(testset, test_embeddings, on = 'id')

    fold = MODELS[0][4]
    train_df = trainset[trainset['fold'] != fold]
    valid_df = trainset[trainset['fold'] == fold]

    X_train = train_df.drop(['id', 'label', 'fold'], inplace = False, axis = 1).values
    X_valid = valid_df.drop(['id', 'label', 'fold'], inplace = False, axis = 1).values 
    X_test  = testset.drop(['id'], inplace = False, axis = 1).values

    y_train = train_df['label'].values
    y_valid = valid_df['label'].values
    
    best_c = None
    best_accuracy = 0
    for c in [C]:
        svm_model = SVC(C = c)
        svm_model.fit(X_train, y_train)

        svm_predictions = svm_model.predict(X_valid)
        accuracy = accuracy_score(y_valid, svm_predictions)
        # print(f"Fold {fold}, [C = {c}], Accuracy: {accuracy}")
        if accuracy > best_accuracy:
            print(f"Fold {fold}, [C = {c}], Accuracy: {accuracy}")
            best_accuracy = accuracy
            best_c = c
            
    best_thresholds.append((best_accuracy, best_c))
    mean_accuracy.append(best_accuracy)
    
    test_predictions = svm_model.predict(X_test)
    votes[:, fold]  = test_predictions
    
print(f"Mean Accuracy: {np.mean(mean_accuracy)}")

final_predictions = []
for i in range(votes.shape[0]):
    values, counts = np.unique(votes[i], return_counts = True)
    index = np.argmax(counts)
    final_predictions.append(values[index])
    
votes = pd.DataFrame(votes.astype(int), columns = [f"vote_{vote}" for vote in range(votes.shape[1])])
votes['id'] = testset['id']

submission = pd.DataFrame(columns = ['id', 'label'])
submission['id']    = testset['id']
submission['label'] = final_predictions
submission['label'] = submission['label'].astype(int)

display(votes)
display(submission)

#  0.765161290322500
# submission.to_csv("sumbission_svm_stage_1_gpu_1_version_42.csv", index = False)
# votes.to_csv("votes_stage_1_gpu_1_version_42.csv", index = False)


# In[8]:


submission.to_csv("sumbission_svm_multi_backbone.csv", index = False)
votes.to_csv("votes_svm_multi_backbone.csv", index = False)


# In[6]:


FOLD_MODELS = [
    [(3, 1, 1, True,  0, "0.774")], # Done
    [(3, 0, 1, True,  1, "0.778")], # Done
    [(3, 1, 1, True,  2, "0.778")], # Done
    [(3, 0, 1, True,  3, "0.771")], # Done
    [(3, 1, 1, True,  4, "0.772")], # Done
    [(3, 1, 1, True,  5, "0.783")], # Done
    [(3, 1, 1, True,  6, "0.778")], # Done
    [(3, 0, 1, False, 7, "0.767")], # Done
    [(3, 1, 1, True,  8, "0.775")], # Done
    [(3, 0, 1, True,  9, "0.768")], # Done
]

votes = np.zeros((5500, len(FOLD_MODELS)))
mean_accuracy = []
best_thresholds = []
for MODELS in FOLD_MODELS:
    trainset = pd.read_csv(PATH_TO_TRAIN_META)
    testset  = pd.read_csv(PATH_TO_TEST_META)

    folds    = pd.read_csv(PATH_TO_FOLDS)
    trainset['fold'] = folds['fold']
    del folds; gc.collect()

    for i, (stage, gpu, version, is_swa, fold, baseline) in enumerate(MODELS):
        print("Stage {}, GPU {}, Model {}, Fold {}, Baseline {}".format(stage, gpu, version, fold, baseline))
        if is_swa:
            train_features  = f'swa_train_stage_{stage}_gpu_{gpu}_version_{version}_fold_{fold}_baseline_{baseline}.csv'
            test_features   = f'swa_test_stage_{stage}_gpu_{gpu}_version_{version}_fold_{fold}_baseline_{baseline}.csv'
        else:
            train_features  = f'train_stage_{stage}_gpu_{gpu}_version_{version}_fold_{fold}_baseline_{baseline}.csv'
            test_features   = f'test_stage_{stage}_gpu_{gpu}_version_{version}_fold_{fold}_baseline_{baseline}.csv'

        train_embeddings_path = os.path.join(PATH_TO_EMBEDDINGS, train_features)
        test_embeddings_path  = os.path.join(PATH_TO_EMBEDDINGS, test_features)

        train_embeddings = pd.read_csv(train_embeddings_path)
        trainset         = pd.merge(trainset, train_embeddings, on = 'id')

        test_embeddings  = pd.read_csv(test_embeddings_path)
        testset          = pd.merge(testset, test_embeddings, on = 'id')

    fold = MODELS[0][4]
    train_df = trainset[trainset['fold'] != fold]
    valid_df = trainset[trainset['fold'] == fold]

    X_train = train_df.drop(['id', 'label', 'fold'], inplace = False, axis = 1).values
    X_valid = valid_df.drop(['id', 'label', 'fold'], inplace = False, axis = 1).values 
    X_test  = testset.drop(['id'], inplace = False, axis = 1).values

    y_train = train_df['label'].values
    y_valid = valid_df['label'].values
    
    best_c = None
    best_accuracy = 0
    for c in [0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:
        svm_model = SVC(C = c)
        svm_model.fit(X_train, y_train)

        svm_predictions = svm_model.predict(X_valid)
        accuracy = accuracy_score(y_valid, svm_predictions)
        if accuracy > best_accuracy:
            print(f"Fold {fold}, [C = {c}], Accuracy: {accuracy}")
            best_accuracy = accuracy
            best_c = c
            
    best_thresholds.append((best_accuracy, best_c))
    svm_model = SVC(C = best_c)
    svm_model.fit(X_train, y_train)

    svm_predictions = svm_model.predict(X_valid)
    accuracy = accuracy_score(y_valid, svm_predictions)
    print(f"Final Fold {fold}, [C = {best_c}], Accuracy: {accuracy}")
        
    test_predictions = svm_model.predict(X_test)
    votes[:, fold]   = test_predictions

print(f"Mean Accuracy: {np.mean(mean_accuracy)}")
    
final_predictions = []
for i in range(votes.shape[0]):
    values, counts = np.unique(votes[i], return_counts = True)
    index = np.argmax(counts)
    final_predictions.append(values[index])
    
votes = pd.DataFrame(votes.astype(int), columns = [f"vote_{vote}" for vote in range(votes.shape[1])])
votes['id'] = testset['id']

submission = pd.DataFrame(columns = ['id', 'label'])
submission['id']    = testset['id']
submission['label'] = final_predictions
submission['label'] = submission['label'].astype(int)

display(votes)
display(submission)

# submission.to_csv("submission_svm_best_folds.csv", index = False)
# votes.to_csv("votes_svm_best_folds.csv", index = False)


# In[7]:


submission.to_csv("submission_svm_best_folds.csv", index = False)
votes.to_csv("votes_svm_best_folds.csv", index = False)


# In[ ]:





# In[ ]:





# In[ ]:




