from IPython.display import clear_output
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from config import GENERATOR_TRAIN, DISCRIMINATOR_TRAIN

DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def imshow(inp):
    """Imshow для тензоров"""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = 0.5
    std = 0.5
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    plt.grid(False)
    
    
def train(models: dict, criterions: dict, train_dl):
    lr = 2e-4
    epochs = 10
    optimizers = {
        "discriminator": torch.optim.Adam(models["discriminator"].parameters(), 
                                          lr=lr, betas=(0.5, 0.999)),
        "generator": torch.optim.Adam(models["generator"].parameters(),
                                      lr=lr, betas=(0.5, 0.999))
    }
    # Losses & scores
    losses_g = []
    losses_d = []
    real_scores = []
    fake_scores = []

    iters_to_show = 100
    iters = 0
    models["discriminator"].train()
    models["generator"].train()
    for epoch in range(epochs):
        loss_d_per_iter = []
        loss_g_per_iter = []
        real_score_per_iter = []
        fake_score_per_iter = []
        for idx, image in enumerate(train_dl):
            image = image.to(DEVICE)
            # Train discriminator

            # Clear discriminator gradients
            optimizers["discriminator"].zero_grad()

            # Freeze generator weights
            for param in models["generator"].parameters():
                param.requires_grad = False
            for param in models["discriminator"].parameters():
                param.requires_grad = True

            # Splitting image into edges and shoes
            image_A = image[:, :3, :, :]
            image_B = image[:, 3:, :, :]

            # Pass real images through discriminator
            real_images = torch.cat([image_A, image_B], dim=1)
            real_preds = models["discriminator"](real_images)
            real_targets = torch.ones(real_preds.shape, device=DEVICE)
            real_loss = criterions["logloss_d"](real_preds, real_targets)
            cur_real_score = torch.mean(torch.sigmoid(real_preds)).item()
            
            # Generate fake images
            fake_B = models["generator"](image_A)

            # Pass fake images through discriminator
            fake_images = torch.cat([image_A, fake_B], dim=1)
            fake_preds = models["discriminator"](fake_images.detach())
            fake_targets = torch.zeros(fake_preds.shape, device=DEVICE)
            fake_loss = criterions["logloss_d"](fake_preds, fake_targets)
            cur_fake_score = torch.mean(torch.sigmoid(fake_preds)).item()

            real_score_per_iter.append(cur_real_score)
            fake_score_per_iter.append(cur_fake_score)

            # Update discriminator weights
            loss_d = (real_loss + fake_loss) * 0.5
            loss_d.backward()
            optimizers["discriminator"].step()
            loss_d_per_iter.append(loss_d.item())


            # Train generator

            # Clear generator gradients
            optimizers["generator"].zero_grad()

            # Freeze discriminator weights
            for param in models["generator"].parameters():
                param.requires_grad = True
            for param in models["discriminator"].parameters():
                param.requires_grad = False
            image_A = image[:, :3, :, :]
            image_B = image[:, 3:, :, :]
            
            # Generate fake images
            fake_B = models["generator"](image_A)
            
            # Try to fool the discriminator
            fake_images = torch.cat([image_A, fake_B], dim=1)
            preds = models["discriminator"](fake_images)
            targets = torch.ones(preds.shape, device=DEVICE)
            fool_loss = criterions["logloss_g"](preds, targets)
            l1_loss = criterions['l1'](fake_B, image_B)
            
            # Update generator weights
            loss_g = fool_loss + 50 * l1_loss
            loss_g.backward()
            optimizers["generator"].step()
            loss_g_per_iter.append(loss_g.item())

            if (idx + 1) % iters_to_show == 0:
                iters += 1
                clear_output()
                # Record losses & scores
                losses_g.append(np.mean(loss_g_per_iter))
                losses_d.append(np.mean(loss_d_per_iter))
                real_scores.append(np.mean(real_score_per_iter))
                fake_scores.append(np.mean(fake_score_per_iter))

                # Save models on each iter
                torch.save(models["generator"].state_dict(), 
                           GENERATOR_TRAIN) # change to your path
                torch.save(models["discriminator"].state_dict(), 
                           DISCRIMINATOR_TRAIN)  # change to your path
                
                # Show generated images
                with torch.no_grad():
                    plt.figure(figsize=(8, 6), dpi=80)
                    for image in train_dl:
                        image = image.to(DEVICE)
                        image_A, image_B = image[:, :3, :, :], image[:, 3:, :, :]
                        fake_B = models["generator"](image_A)
                        imshow(torch.cat([image_A[0].cpu(), image_B[0].cpu(), 
                                          fake_B[0].cpu()], dim=2))
                        plt.show()
                        break

                # Log losses & scores (last iter)
                print(f'Epoch: {epoch + 1}, All: {epochs}')
                print(f'Iter: {iters}, All: {len(train_loader) // iters_to_show}')
                print(f'Generator loss: {losses_g[-1]}')
                print(f'Discriminator loss: {losses_d[-1]}')
                print(f'Score on real: {real_scores[-1]}')
                print(f'Score on fake: {fake_scores[-1]}')
    
    return losses_g, losses_d, real_scores, fake_scores
