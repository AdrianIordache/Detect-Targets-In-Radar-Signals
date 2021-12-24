import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display

STAGES_GPUS_VERSIONS = [(3, 0, 1, 'N'), (3, 0, 1, 'SWA'), (3, 1, 1, 'N'), (3, 1, 1, 'SWA')]

general_votes = pd.DataFrame()
general_preds = pd.DataFrame()

for model, (stage, gpu, version, types) in enumerate(STAGES_GPUS_VERSIONS):
    if types == 'N':
        path_to_submission = f'ensambles/original-subs/submission_stage_{stage}_gpu_{gpu}_version_{version}.csv'
        path_to_votes      = f'ensambles/votes/votes_stage_{stage}_gpu_{gpu}_version_{version}.csv'
    elif types == 'S':
        path_to_submission = f'embeddings/submissions/submission_svm_stage_{stage}_gpu_{gpu}_version_{version}.csv'
        path_to_votes      = f'embeddings/votes/votes_svm_stage_{stage}_gpu_{gpu}_version_{version}.csv'
    elif types == 'SWA':
        path_to_submission = f'ensambles/original-subs/swa_submission_stage_{stage}_gpu_{gpu}_version_{version}.csv'
        path_to_votes      = f'ensambles/votes/swa_votes_stage_{stage}_gpu_{gpu}_version_{version}.csv'

    submission = pd.read_csv(path_to_submission)
    votes      = pd.read_csv(path_to_votes)

    if model == 0:
        general_votes['id'] = submission['id']
        general_preds['id'] = submission['id']

    submission = submission.rename(columns = {"label": f"model_{model}_label"})
    votes      = votes.rename(columns = {column : f"model_{model}_{column}" for column in votes.columns.tolist() if column != 'id'})

    general_preds = pd.merge(general_preds, submission, on = 'id')
    general_votes = pd.merge(general_votes, votes, on = 'id')

# Stage-3 ALL 4 (Best-Folds)
votes = general_votes[['model_3_vote_0', 'model_1_vote_1', 'model_3_vote_2', 'model_1_vote_3', 'model_3_vote_4', 'model_3_vote_5', 'model_3_vote_6', 'model_0_vote_7', 'model_3_vote_8', 'model_1_vote_9', 'id']]

svm_submission = pd.read_csv('embeddings/submissions/submission_svm_best_folds.csv')
svm_votes      = pd.read_csv('embeddings/votes/votes_svm_best_folds.csv')

general_preds = pd.merge(general_preds, svm_submission, on = 'id')
votes         = pd.merge(votes, svm_votes, on = 'id')

display(general_preds)
display(votes)

display(general_preds.corr())
display(votes.corr())

votes = votes.drop('id', axis = 1, inplace = False).values

print(votes.shape)

candidates  = []
differences = []
first_candidate  = []
second_candidate = []
both_candidates  = []

count_ones = 0
final_predictions = []
for i in range(votes.shape[0]):
    values, counts = np.unique(votes[i], return_counts = True)

    if len(values) == 1: 
        count_ones += 1

    index = np.argmax(counts)
    final_predictions.append(values[index])

print(f"All voted: {count_ones}")

if 0:
    submission = pd.DataFrame(columns = ['id', 'label'])
    submission['id']    = general_votes['id']
    submission['label'] = final_predictions
    submission['label'] = submission['label'].astype(int)

    submission.to_csv("ensambles/ensamble_voting_17.csv", index = False)
    display(submission)


# (0, 0, 0, 'N')  -> 0.80000
# (1, 1, 42, 'N') -> 0.80072
# (0, 0, 0, 'N') + (1, 1, 42, 'N') -> 0.80872

# (1, 0, 43, 'N') -> 0.79636
# (0, 0, 0, 'N') + (1, 1, 42, 'N') + (1, 0, 43, 'N') -> 0.80072

# (1, 1, 43, 'N') -> 0.80000
# (0, 0, 0, 'N') + (1, 1, 42, 'N') + (1, 1, 43, 'N') -> 0.80581

# (1, '1+1', '42+43', 'S') -> 0.80363
# (1, 1, 42, 'S') + (1, 1, 43, 'S') + (1, '1+1', '42+43', 'S') -> 0.80509

# (1, 1, 42, 'N') + (1, 1, 43, 'N') + (1, 1, 42, 'S') + (1, 1, 43, 'S') + (1, '1+1', '42+43', 'S') -> 0.80218

# (0, 0, 0, 'N')  + (1, 1, 42, 'N') + (1, '1+1', '42+43', 'S') -> 0.80872
# (0, 0, 0, 'N')  + (1, '1+1', '42+43', 'S') -> 0.79781
# (1, 1, 42, 'N') + (1, '1+1', '42+43', 'S') -> 0.80363

# (3, 0, 0, 'N'), (3, 0, 0, 'SWA') (Best Folds) -> 0.81018
# (3, 1, 0, 'N'), (3, 1, 0, 'SWA') (Best Folds) -> 0.81018

# (3, 0, 0, 'N'), (3, 0, 0, 'SWA'), (3, 1, 0, 'N'), (3, 1, 0, 'SWA') (All Folds)  -> 0.81090
# (3, 0, 0, 'N'), (3, 0, 0, 'SWA'), (3, 1, 0, 'N'), (3, 1, 0, 'SWA') (Best Folds) -> 0.80800

# (3, 0, 1, 'N'), (3, 0, 1, 'SWA') (All Folde)  -> 0.81090
# (3, 0, 1, 'N'), (3, 0, 1, 'SWA') (Best Folde) -> 0.80145

# (3, 0, 1, 'N'), (3, 0, 1, 'SWA'), (3, 1, 1, 'N'), (3, 1, 1, 'SWA') (All-Folds)  -> 0.80509
# (3, 0, 1, 'N'), (3, 0, 1, 'SWA'), (3, 1, 1, 'N'), (3, 1, 1, 'SWA') (Best-Folds) -> 0.80727
