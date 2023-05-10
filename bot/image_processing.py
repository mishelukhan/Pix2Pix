import torch 
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np

async def process(model, path2image, path2shoes):
    '''
    image: [3, W, H]
    Need to resize to [3, 256, 256] and convert to RGB.
    '''
    with torch.no_grad():
        image = Image.open(path2image).convert('RGB')
        image = image.resize((256, 256),Image.ANTIALIAS)
        mean = 0.5
        std = 0.5
        transform_general = transforms.Compose([
            transforms.Lambda(lambda x: np.array(np.array(x) / 255, dtype='float32')),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x - mean) / std)
        ])
        image = transform_general(image)
        output = model(image[None, :, :, :])
        output = std * output + mean
        output = np.clip(output, 0, 1)
        save_image(output[0], path2shoes)