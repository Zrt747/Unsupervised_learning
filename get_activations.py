import cv2
import torch.nn.functional as F
import numpy as np
import torch
import torchvision.transforms as transforms

def min_max(x):
    return (x - np.min(x))/(np.max(x) - np.min(x))

def get_activation(activation, img):
    # w,h = img.size
    h,w = img.size

    # Normalize the Grad-CAM
    grad_cam = torch.nn.functional.relu(activation)
    grad_cam = torch.nn.functional.interpolate(grad_cam.unsqueeze(0), size=(w,h), mode='bilinear', align_corners=False)
    grad_cam = grad_cam.detach().squeeze().cpu().numpy()
    grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())
    return grad_cam

def get_all_activations(im,model):
    w,h = im.size
    total_activation = np.zeros(shape=(h,w))
    tensor_im = transforms.ToTensor()(im)
    with torch.no_grad():
        output = model.P2.retinanet.backbone(tensor_im)
    for i, (name,data) in enumerate(output.items()):
        total_activation += get_activation(data.mean(1), im)
    # return min_max(total_activation)
    return total_activation

def get_one_activations(image,model, layer = 'p7'):
    w,h = image.size
    total_activation = np.zeros(shape=(h,w))
    tensor_im = transforms.ToTensor()(image)
    with torch.no_grad():
        output = model.P2.retinanet.backbone(tensor_im)
    # return min_max(get_activation(output[layer].mean(1), image))
    return get_activation(output[layer].mean(1), image)
