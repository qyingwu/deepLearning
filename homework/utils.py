from PIL import Image

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from os import path
import csv

LABEL_NAMES = ['background', 'kart', 'pickup', 'nitro', 'bomb', 'projectile']

class SuperTuxDataset(Dataset):
    """
    WARNING: Do not perform data normalization here. 
    """
    def __init__(self, dataset_path):
        """
        Your code here
        Hint: Use your solution (or the master solution) to HW1
        """
        # raise NotImplementedError('SuperTuxDataset.__init__')
        self.data = []
        toTensor = transforms.ToTensor()
        with open(path.join(dataset_path, 'labels.csv'), newline='') as csvfile:
            reader = csv.reader(csvfile)
            for fileName, label, _ in reader:
                if label in LABEL_NAMES:
                    image = Image.open(path.join(dataset_path, fileName))
                    imageTensor = toTensor(image)
                    labelID = LABEL_NAMES.index(label)
                    self.data.append( (imageTensor, labelID) )

    def __len__(self):
        """
        Your code here
        """
        # raise NotImplementedError('SuperTuxDataset.__len__')
        return len(self.data)

    def __getitem__(self, idx):
        """
        Your code here
        return a tuple: img, label
        """
        # raise NotImplementedError('SuperTuxDataset.__getitem__')
        return self.data[idx]


def load_data(dataset_path, num_workers=0, batch_size=128):
    dataset = SuperTuxDataset(dataset_path)
    return DataLoader(dataset, num_workers=num_workers, batch_size=batch_size, shuffle=True, drop_last=True)


def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(labels)
    return outputs_idx.eq(labels).float().mean()
