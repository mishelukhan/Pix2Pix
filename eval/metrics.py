import numpy as np
import torch
from torchvision import transforms
from PIL import Image
from scipy import linalg
from torch.nn.functional import adaptive_avg_pool2d
from torch.utils.data import Dataset, DataLoader
from pytorch_fid.inception import InceptionV3
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.manifold import TSNE
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm


class ImagePathDataset(Dataset):
    def __init__(self, files, transforms=None):
        self.files = files
        self.transforms = transforms

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        image = Image.open(path)
        image.load()
        if self.transforms is not None:
            image = self.transforms(image)
        return image


def get_activations(files, model, batch_size=50, dims=2048, device='cpu',
                    num_workers=1):
    model.eval()

    dataset = ImagePathDataset(files, transforms=transforms.ToTensor())
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            drop_last=False, num_workers=num_workers)

    pred_arr = np.empty((len(files), dims))

    start_idx = 0

    for batch in tqdm(dataloader):
        batch = batch.to(device)

        with torch.no_grad():
            pred = model(batch)[0]
            
        if pred.size(2) != 1 or pred.size(3) != 1:
            pred = adaptive_avg_pool2d(pred, output_size=(1, 1))

        pred = pred.squeeze(3).squeeze(2).cpu().numpy()

        pred_arr[start_idx:start_idx + pred.shape[0]] = pred

        start_idx = start_idx + pred.shape[0]

    return pred_arr


def calculate_activation_statistics(files, model, batch_size=50, dims=2048,
                                    device='cpu', num_workers=1):
    
    act = get_activations(files, model, batch_size, dims, device, num_workers)
    mu = np.mean(act, axis=0)
    sigma = np.cov(act, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    # Count difference in means
    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

def FID_metric(files_real, files_fake, device='cpu', dims=2048):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    mu_real, sigma_real = calculate_activation_statistics(files_real, model)
    mu_fake, sigma_fake = calculate_activation_statistics(files_fake, model)
    fid_score = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    return fid_score
    

def LOO_score(files_real, files_fake, device='cpu', dims=2048):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    
    preds_real = get_activations(files_real, model)
    preds_fake = get_activations(files_fake, model)
    preds_all = np.vstack([preds_real, preds_fake])
    
    target_real = np.ones(preds_real.shape[0])
    target_fake = np.zeros(preds_fake.shape[0])
    target_all = np.hstack([target_real, target_fake])
    
    all_embedded = TSNE(n_components=2, perplexity=3).fit_transform(preds_all)
    one_nn = KNeighborsClassifier(n_neighbors=1)
    scores = cross_val_score(one_nn, all_embedded, target_all, cv=LeaveOneOut())
    
    return scores.mean()


def TSNE_plot(files_real, files_fake, device='cpu', dims=2048):
    block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[dims]
    model = InceptionV3([block_idx]).to(device)
    
    preds_real = get_activations(files_real, model)
    preds_fake = get_activations(files_fake, model)
    preds_all = np.vstack([preds_real, preds_fake])
    
    target_real = np.ones(preds_real.shape[0])
    target_fake = np.zeros(preds_fake.shape[0])
    target_all = np.hstack([target_real, target_fake])
    
    all_embedded = TSNE(n_components=2, perplexity=3).fit_transform(preds_all)
    sns.scatterplot(x=all_embedded[:, 0], y=all_embedded[:, 1], hue = target_all)
    plt.show()
    

def plotting(losses_g, losses_d, real_scores, fake_scores):
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_figheight(5)
    fig.set_figwidth(15)
    
    ax1.plot(np.arange(0, len(losses_g), 1), losses_g, 
             label='Generator loss')
    ax1.plot(np.arange(0, len(losses_d), 1), losses_d, 
             label='Discriminator loss')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Iter')
    ax1.grid()
    ax1.legend()
    
    ax2.plot(np.arange(0, len(real_scores), 1), 
             real_scores, label='Accuracy on real images')
    ax2.plot(np.arange(0, len(fake_scores), 1), 
             fake_scores, label='Accuracy on fake images')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Iter')
    ax2.grid()
    ax2.legend()
    
    plt.show()
    