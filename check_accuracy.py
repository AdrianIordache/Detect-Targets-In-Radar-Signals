from utils import *

MODELS = [
	(3, 0, 1, False, 
	[(0, '0.77'), (1, '0.77'), (2, '0.77'), (3, '0.76'), (4, '0.75'), 
	 (5, '0.77'), (6, '0.77'), (7, '0.77'), (8, '0.77'), (9, '0.76')]),

	(3, 0, 1, True, 
	[(0, '0.76'), (1, '0.77'), (2, '0.77'), (3, '0.76'), (4, '0.75'), 
	 (5, '0.77'), (6, '0.76'), (7, '0.75'), (8, '0.77'), (9, '0.76')]),


	(3, 1, 0, False, 
	[(0, '0.77'), (1, '0.77'), (2, '0.77'), (3, '0.77'), (4, '0.75'), 
	 (5, '0.77'), (6, '0.75'), (7, '0.77'), (8, '0.77'), (9, '0.76')]),

	(3, 1, 0, True, 
	[(0, '0.77'), (1, '0.78'), (2, '0.77'), (3, '0.75'), (4, '0.75'), 
	 (5, '0.78'), (6, '0.76'), (7, '0.78'), (8, '0.77'), (9, '0.75')]),


	(3, 1, 1, False, 
	[(0, '0.77'), (1, '0.77'), (2, '0.77'), (3, '0.77'), (4, '0.76'), 
	 (5, '0.78'), (6, '0.77'), (7, '0.77'), (8, '0.77'), (9, '0.76')]),

	(3, 1, 1, True, 
	[(0, '0.77'), (1, '0.77'), (2, '0.77'), (3, '0.76'), (4, '0.77'), 
	 (5, '0.77'), (6, '0.77'), (7, '0.76'), (8, '0.77'), (9, '0.76')]),

]

RENAME = True
for (stage, gpu, model, is_swa, folds) in MODELS:
	path = f'models/stage-{stage}/gpu-{gpu}/model_{model}/'

	all_labels = []
	all_predictions = []
	for (fold, accuracy) in folds:
		model_name = f'model_{model}_name_swin_large_patch4_window12_384_in22k_fold_{fold}_accuracy_{accuracy}.pth'
		
		if is_swa == True:
			model_name = 'swa_' + model_name	

		path_to_model = os.path.join(path, model_name)
		states = torch.load(path_to_model, map_location = torch.device('cpu'))

		labels      = states['oof_labels']
		predictions = states['oof_proba']

		new_accuracy = accuracy_score(labels, predictions)
		print(f"Stage {stage}, GPU {gpu}, Model {model}, SWA {is_swa}, Fold {fold}, Accuracy {accuracy}, New Accuracy {new_accuracy:.3f}")

		if RENAME:
			if is_swa == True:
				os.rename(path_to_model, os.path.join(path, f"swa_model_{model}_name_swin_large_patch4_window12_384_in22k_fold_{fold}_accuracy_{new_accuracy:.3f}.pth"))
			else:
				os.rename(path_to_model, os.path.join(path, f"model_{model}_name_swin_large_patch4_window12_384_in22k_fold_{fold}_accuracy_{new_accuracy:.3f}.pth"))

		all_labels.extend(labels)
		all_predictions.extend(predictions) 
	
	oof_accuracy = accuracy_score(all_labels, all_predictions)
	print(f"Stage {stage}, GPU {gpu}, Model {model}, SWA {is_swa}, OOF: {oof_accuracy:.4f}")
