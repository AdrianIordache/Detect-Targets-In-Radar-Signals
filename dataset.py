from utils import *

class RadarSignalsDataset(Dataset):
    def __init__(self, dataset: pd.DataFrame = None, train: bool = True, transform = None):
        super().__init__()
        self.dataset   = dataset
        self.train     = train
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image_path = self.dataset['path'].iloc[idx]
        image = cv2.imread(image_path)
        
        if self.transform:
            augmented = self.transform(image = image)
            image = augmented['image']

        if self.train:
            label = self.dataset['label'].iloc[idx]
            label = torch.tensor(label)
            return image, label

        return image