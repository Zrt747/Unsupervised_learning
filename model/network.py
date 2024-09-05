import torch
import torch.nn as nn
from .pipelines import Pipeline1,Pipeline2
from .utils import random_crop_batch
import torchvision.transforms as transforms

class Unsupervised_Object_Detection(nn.Module):
    def __init__(self,):
        super(Unsupervised_Object_Detection, self).__init__() 
        self.P1 = Pipeline1()
        self.P2 = Pipeline2()

    def __call__(self, image, *args: torch.Any, **kwds: torch.Any) -> torch.Any:

        # need to bullet proof here ..
        if not isinstance(image, torch.Tensor):
            image = transforms.ToTensor()(image)
        while image.ndim < 4:
            image = image.unsqueeze(0) # needs to be 4d!
        im_size = (image.size(2),image.size(3))

        xi, focal_point = random_crop_batch(image)
        
        # pipeline 1
        zi = self.P1(xi).squeeze()

        # pipline 2
        zj, za = self.P2(image,zi)   

        return zi, zj, za