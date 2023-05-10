from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms
import numpy as np
import torch

RESCALE_SIZE = 286
CROPPED_SIZE = 256
BATCH_SIZE = 4

class Pix2pixDataset(Dataset):
    def __init__(self, files: list, mode: bool):
        self.files = files
        self.mode = mode
        self.len_ = len(self.files)
    
    def __len__(self):
        return self.len_
    
    def load_sample(self, file: str):
        image = Image.open(file)
        image.load()
        return image

    def __getitem__(self, index):
        image = self.load_sample(self.files[index])
        image = self._prepare_sample(image)
        return image

    def _prepare_sample(self, image):
        width, height = image.size
        left = 0
        right = width // 2
        image_A = image.crop((left, 0, right, height))
        left = width // 2
        right = width
        image_B = image.crop((left, 0, right, height))
        if self.mode == 'train':
            image_A = image_A.resize((RESCALE_SIZE, RESCALE_SIZE))
            image_B = image_B.resize((RESCALE_SIZE, RESCALE_SIZE))
        mean = 0.5
        std = 0.5
        transform_general = transforms.Compose([
            transforms.Lambda(lambda x: np.array(np.array(x) / 255, dtype='float32')),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x - mean) / std)
        ])
        image_A = transform_general(image_A)
        image_B = transform_general(image_B)
        stacked = torch.cat((image_A, image_B), dim=0)
        transforms_train = transforms.Compose([
            transforms.RandomCrop((CROPPED_SIZE, CROPPED_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5)
        ])
        if self.mode == 'train':
            stacked = transforms_train(stacked)

        return stacked