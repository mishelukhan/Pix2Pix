import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from config import ROOT_DIR

DEVICE = torch.device('cpu')

def evaluate(model, test_dl):
    idx = 0
    root_dir = ROOT_DIR # change to path with your project
    real_images_dir = os.path.join(root_dir, 'real_images')
    if not os.path.exists(real_images_dir):
        os.mkdir(real_images_dir)
    fake_images_dir = os.path.join(root_dir, 'fake_images')
    if not os.path.exists(fake_images_dir):
        os.mkdir(fake_images_dir)
    with torch.no_grad():
        for image in test_dl:
            image = image.to(DEVICE)
            image_A = image[:, :3, :, :]
            image_B = image[:, 3:, :, :]
            fake_B = model(image_A)

            # Unnormalized
            mean = 0.5
            std = 0.5
            image_A = std * image_A + mean
            image_A = np.clip(image_A, 0, 1)
            image_B = std * image_B + mean
            image_B = np.clip(image_B, 0, 1)
            fake_B = std * fake_B + mean
            fake_B = np.clip(fake_B, 0, 1)

            nums_images = image.shape[0]
            for each in range(nums_images):
                name_pair = 'test' + str(idx)
                idx += 1
                path_test = os.path.join(root_dir, name_pair)
                if not os.path.exists(path_test):
                    os.mkdir(path_test)
                path_A = os.path.join(path_test, 'img_A.png')
                save_image(image_A[each], path_A)
                path_B = os.path.join(path_test, 'img_B.png')
                save_image(image_B[each], path_B)
                path_fake = os.path.join(path_test, 'img_fake.png')
                save_image(fake_B[each], path_fake)

                name_real = 'real' + str(idx) + '.png'
                path_real = os.path.join(real_images_dir, name_real)
                name_fake = 'fake' + str(idx) + '.png'
                path_fake = os.path.join(fake_images_dir, name_fake)
                save_image(image_B[each], path_real)
                save_image(fake_B[each], path_fake)
                    
            