import albumentations as A 

configs = {
	"GPU-0":{
		"model_config": [('swin_large_patch4_window12_384_in22k', 384)],
		"optimizers": [('RangerLars', (5, 32), 3100)],
		"lrs": [7e-6, 1e-5, 3e-5, 5e-5, 7e-5, 1e-4, 3e-4],
		"schedulers": [
			{
				'scheduler_name': 'CosineAnnealingWarmRestarts',
				'T_0': 200,
				'T_mult': 2,
				'min_lr': 1e-6
			}, 
			{
				'scheduler_name': 'CosineAnnealingWarmRestarts',
				'T_0': 500,
				'T_mult': 5,
				'min_lr': 1e-6
			}, 
			{
				'scheduler_name': 'OneCycleLR',
				'no_batches': 3100,
				'epochs': 20,
				'max_lr': 1e-3
			},
		],
		"augmentations": [
			[],
			[
				A.HorizontalFlip(p = 0.5),
				A.VerticalFlip(p = 0.5),
				A.RandomRotate90(p = 0.5),
			]
		]
	},
	"GPU-1":{
		"model_config": [('swin_large_patch4_window12_384_in22k', 384)],
		"optimizers": [('AdamW', (5, 32), 2480)],
		"lrs": [7e-6, 1e-5, 3e-5, 5e-5, 7e-5, 1e-4, 3e-4],
		"schedulers": [
			{
				'scheduler_name': 'CosineAnnealingWarmRestarts',
				'T_0': 200,
				'T_mult': 2,
				'min_lr': 1e-6
			}, 
			{
				'scheduler_name': 'CosineAnnealingWarmRestarts',
				'T_0': 500,
				'T_mult': 5,
				'min_lr': 1e-6
			}, 
			{
				'scheduler_name': 'OneCycleLR',
				'no_batches': 2480,
				'epochs': 20,
				'max_lr': 1e-3
			},
		],
		"augmentations": [
			[],
			[
				A.HorizontalFlip(p = 0.5),
				A.VerticalFlip(p = 0.5),
				A.RandomRotate90(p = 0.5),
			]
		],
	}
}
