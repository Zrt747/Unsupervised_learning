import torch
import torch.nn as nn
import random
import torchvision.transforms.functional as F
from torchvision.models import resnet50
from torchvision.models.detection import retinanet_resnet50_fpn
import numpy as np
from functools import partial
from torchvision.models.detection.retinanet import RetinaNetClassificationHead


class Pipeline1(nn.Module):
    def __init__(self, projection_dim=256):
        super(Pipeline1, self).__init__()
        """
        Initializes the Pipeline1 class with a ResNet encoder and a projection head.

        Parameters:
        projection_dim (int): Dimension of the projection head output.
        """
        self.resnet = resnet50(weights='DEFAULT')
        
        # Freeze all layers except layer 3 and 4
        for name, param in self.resnet.named_parameters():
            if "layer3" not in name and "layer4" not in name:
                param.requires_grad = False

        self.resnet.eval()

        self.projection_head = torch.nn.Linear(1000, projection_dim)  # Adjust input dimension as per ResNet output

    def __call__(self, x):
        """
        Applies random augmentations (HFlip or VFlip) to the input tensor, 
        then passes it through a ResNet encoder network and a projection head.

        Parameters:
        x (torch.Tensor): Input image tensor.

        Returns:
        torch.Tensor: The embedding generated after the projection head.
        """

        # Apply random horizontal or vertical flip
        x = random_flip(x)

        with torch.no_grad():
            r = self.resnet(x)

        z = self.projection_head(r)
        
        return z


class Pipeline2(nn.Module):
    def __init__(self):
        super(Pipeline2, self).__init__()
        """
        Initializes the Pipeline2 class with a RetinaNet encoder and a projection head.

        Parameters:
        projection_dim (int): Dimension of the projection head output.
        """
        self.retinanet = retinanet_resnet50_fpn(weights='DEFAULT')

        # change n classes
        num_classes = 5 # 4 class + background
        in_features = self.retinanet.head.classification_head.cls_logits.in_channels
        num_anchors = self.retinanet.head.classification_head.num_anchors

        self.retinanet.head.classification_head = RetinaNetClassificationHead(
            in_channels=256,  # This is typically the default in-channel size for FPN layers
            num_anchors=num_anchors,
            num_classes=num_classes,
            norm_layer=partial(torch.nn.GroupNorm, 32)
        )
        # self.retinanet.eval()
        self.retinanet.train()

    def random_flip(self, x):
        """
        Applies random horizontal or vertical flip to the input tensor.
        """
        if random.random() > 0.5:
            x = F.hflip(x)
        else:
            x = F.vflip(x)
        return x

    def positive_pair(self, fpn_output, pipe1_emb):
        """
        Generate a positive pair by selecting the feature vector closest to the center.
        """

        fpn_output_np = fpn_output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        h, w, c = fpn_output_np.shape
        central_feature = pipe1_emb

        similarities = []
        for y in range(h):
            for x in range(w):
                feature = fpn_output_np[y, x]
                similarity = np.dot(central_feature, feature) / (np.linalg.norm(central_feature) * np.linalg.norm(feature)) # cosine..
                similarities.append((similarity, y, x))

        similarities.sort(reverse=True, key=lambda x: x[0])
        most_similar_feature = similarities[0]
        positive_counterpart = fpn_output_np[most_similar_feature[1], most_similar_feature[2]]

        return torch.tensor(positive_counterpart, device=fpn_output.device)

    def neg_anchor(self, fpn_output, num_negatives=10):
        """
        Generate anchor negatives by randomly selecting other locations within the image.
        """
        fpn_output_np = fpn_output.squeeze(0).permute(1, 2, 0).cpu().numpy()
        h, w, c = fpn_output_np.shape

        np.random.seed(0)  # For reproducibility
        negative_indices = np.random.choice(h * w, num_negatives, replace=False)
        anchor_negatives = [fpn_output_np[idx // w, idx % w] for idx in negative_indices]

        return torch.tensor(anchor_negatives, device=fpn_output.device)

    def __call__(self, x,pipe1_emb):
        """
        Applies random augmentations (HFlip or VFlip) to the input tensor, 
        then passes it through a RetinaNet encoder network, and performs positive and negative pair operations.

        Parameters:
        x (torch.Tensor): Input image tensor.

        Returns:
        torch.Tensor: The embedding generated after the projection head.
        """

        pipe1_emb = pipe1_emb.detach().numpy()

        # Apply random horizontal or vertical flip
        x = self.random_flip(x)

        with torch.no_grad():
            features = self.retinanet.backbone(x)
        
        # Choose a specific FPN layer output (e.g., '0')
        fpn_output = features['0']

        zj = self.positive_pair(fpn_output, pipe1_emb)
        za = self.neg_anchor(fpn_output) #  double check this
        zj = zj * za.mean(dim=0)
        
        return zj, za



def random_flip(x):
    x_augmented = []
    for img in x:
        if random.random() > 0.5:
            img = F.hflip(img)
        else:
            img = F.vflip(img)
        x_augmented.append(img)
    
    x = torch.stack(x_augmented)
    return x


    # Function to extract FPN embeddings
def get_fpn_embeddings(model, x):
    
    # Register hooks to capture the outputs of FPN layers
    fpn_outputs = []
    def hook(module, input, output):
        fpn_outputs.append(output)
    
    # Get FPN layers and register hooks
    layers = [model.backbone.fpn.inner_blocks[i] for i in range(5)]
    hooks = [layer.register_forward_hook(hook) for layer in layers]
    
    # Forward pass through the model
    with torch.no_grad():
        model(x)
    
    # Remove hooks
    for h in hooks:
        h.remove()
    
    return fpn_outputs