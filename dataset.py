import glob
import numpy as np
import torch, os, pdb
import random as rn
from torch.utils import data
from torchvision import transforms
from torch.utils.data import Sampler
from torch.utils.data import DataLoader
import torch.multiprocessing
from PIL import Image
from scipy.io import loadmat

class SubsetSampler(Sampler):
    r"""Samples elements sequentially from a given list of indices, without replacement.

    Arguments:
        indices (sequence): a sequence of indices
    """

    def __init__(self, indices):
        self.indices = indices

    def __iter__(self):
        return (self.indices[i] for i in range(len(self.indices)))

    def __len__(self):
        return len(self.indices)


class MatLoader(DataLoader):

    def __init__(self, X, batch_size, num_workers, img_size=256, crop_size=128, pin_memory=True, mode='Train'):

        self.faces = dataset_h5(X, batch_size, mode=mode, crop_size=crop_size)
        sampler = SubsetSampler(self.faces.indices)
        shuffle=True if mode=='Train' else False

        super().__init__(dataset=self.faces,
                        
                         sampler=sampler,
                         num_workers=num_workers,
                         pin_memory=pin_memory
                         )

class dataset_h5(torch.utils.data.Dataset):
    def __init__(self, X, img_size=256, crop_size=128, width=4, root='/nvme2/DFC_Images/', mode='Train', marginal=40):
        super(dataset_h5, self).__init__()

        self.root = root
        self.fns = X
        self.n_images = len(self.fns)
        self.indices = np.array(range(self.n_images))

        self.mode=mode
        self.crop_size=crop_size
        self.img_size=img_size
        self.marginal = marginal
        self.width = width
    
        
    def __getitem__(self, index):
        data, label = {}, {}
        
        fn = os.path.join(self.root, self.fns[index])

#         x=loadmat(fn)
#         x=x[list(x.keys())[-1]]
        x = np.load(fn)
        x = x.astype(np.float32)
        xmin = np.min(x)
        xmax = np.max(x)
        
        if self.mode=='Train':
            # Random crop
            shifting_h = (self.img_size-self.crop_size) -1
            shifting_w =  (self.img_size-self.width) - self.marginal-1
            xim, yim = rn.randint(0, shifting_w), rn.randint(0, shifting_h)
            h = yim+self.crop_size
            xx = []
            for k in range(self.marginal):
                y = x[yim:h, xim+k:xim+self.width+k, :]
                # Random flip
                if rn.random()>0.5:
                    y = y[::-1,:,:]
                if rn.random()>0.5:
                    y = y[:,::-1,:]
                
                y = torch.from_numpy(y.copy())
                xx.append(y)
            x = torch.stack(xx)
        else:
            xx = []
            for k in range(0, 256, self.width):
                y = x[:, k:k+self.width, :]
                y = torch.from_numpy(y.copy())
                xx.append(y)
            x = torch.stack(xx)
                
        
        if xmin == xmax:
            return np.zeros((self.marginal, 128, 4, 172))
        x = (x-xmin) / (xmax-xmin)
        
        return x, fn

    def __len__(self):
        return self.n_images