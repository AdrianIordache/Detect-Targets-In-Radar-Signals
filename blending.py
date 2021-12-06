import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from IPython.display import display

# (0, 0, 0)  -> 0.80000
# (1, 1, 42) -> 0.80072
# (1, 0, 43) -> 0.79636
# (0, 0, 0) + (1, 1, 42) -> 0.80872
# (0, 0, 0) + (1, 1, 42) + (1, 0, 43) -> 0.80072
# (1, 1, 43) -> 0.80000
# (0, 0, 0) + (1, 1, 42) + (1, 1, 43) -> 0.80581

STAGES_GPUS_VERSIONS = [(0, 0, 0), (1, 1, 42), (1, 1, 43)]

general_votes = pd.DataFrame()
general_preds = pd.DataFrame()

for model, (stage, gpu, version) in enumerate(STAGES_GPUS_VERSIONS):
    path_to_submission = f'ensambles/original-subs/submission_stage_{stage}_gpu_{gpu}_version_{version}.csv'
    path_to_votes      = f'ensambles/votes/votes_stage_{stage}_gpu_{gpu}_version_{version}.csv'

    submission = pd.read_csv(path_to_submission)
    votes      = pd.read_csv(path_to_votes)

    if model == 0:
        general_votes['id'] = submission['id']
        general_preds['id'] = submission['id']

    submission = submission.rename(columns = {"label": f"model_{model}_label"})
    votes      = votes.rename(columns = {column : f"model_{model}_{column}" for column in votes.columns.tolist() if column != 'id'})

    general_preds = pd.merge(general_preds, submission, on = 'id')
    general_votes = pd.merge(general_votes, votes, on = 'id')

display(general_preds)
display(general_votes)

display(general_preds.corr())
display(general_votes.corr())

votes = general_preds.drop('id', axis = 1, inplace = False).values

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
    else:
        candidates.append(len(values))
        differences.append(np.abs(values[0] - values[1]))
        first_candidate.append(values[0])
        second_candidate.append(values[1])
        both_candidates.append((values[0], values[1]))

    index = np.argmax(counts)
    final_predictions.append(values[index])

print(f"All voted: {count_ones}")

# print(candidates)
# print(differences)

# plt.bar(*np.unique(differences, return_counts = True))
# plt.show()

# plt.bar(*np.unique(first_candidate, return_counts = True))
# plt.show()

# plt.bar(*np.unique(second_candidate, return_counts = True))
# plt.show()

# print(both_candidates)

# print(*np.unique(both_candidates, return_counts = True))

# plt.bar(*np.unique(final_predictions, return_counts = True))
# plt.show()

if 0:
    submission = pd.DataFrame(columns = ['id', 'label'])
    submission['id']    = general_votes['id']
    submission['label'] = final_predictions
    submission['label'] = submission['label'].astype(int)

    submission.to_csv("ensambles/ensamble_voting_2.csv", index = False)
    display(submission)