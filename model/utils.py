import torch
from PIL import Image
import random
import numpy as np
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import math
import matplotlib.pyplot as plt

def random_crop_batch(images):
    """
    Randomly crops a batch of images.

    Parameters:
    images (list of PIL Images): A list of images to be cropped.

    Returns:
    torch.Tensor: A tensor containing the cropped images.

    The function performs the following steps:
    1. Calculates the width (W) and height (H) of the images.
    2. Determines the maximum dimension (max_dim) between width and height.
    3. Sets a random crop size between 10% and 25% of max_dim.
    4. For each image in the batch:
       a. Selects a random top-left corner (x, y) for the crop.
       b. Crops the image using the selected corner and crop size.
       c. Appends the cropped image to the list of cropped_images.
    5. Stacks the cropped images into a tensor and returns it.
    """    
    cropped_images = []
    W, H = _imsize(images)
    max_dim = max(W, H)
    crop_size = random.uniform(0.10 * max_dim, 0.25 * max_dim)
    crop_size = int(crop_size)

    for image in images:
        x = random.randint(0, W - crop_size)
        y = random.randint(0, H - crop_size)
        
        # cropped_image = image.crop((x, y, x + crop_size, y + crop_size))
        cropped_image = F.crop(image, y, x, crop_size, crop_size)
        cropped_images.append(cropped_image)

        focal_point = (y + crop_size//2, x + crop_size//2)
    return torch.stack(cropped_images), focal_point

def _imsize(images): 
    if images.ndim == 2:
        W, H = images.size()
    elif images.ndim == 3:
        _,W, H = images.size()
    elif images.ndim == 4:
        _,_,W, H = images.size()
    else:
        raise ValueError(f'What the heck? batch of images must between 2-4 dimensions')
    return W,H


def random_rotate_image(image):
    original_width, original_height = image.size

    # Generate a random angle for rotation
    angle = random.uniform(0, 360)

    # Rotate the image
    rotated_image = image.rotate(angle, resample=Image.BICUBIC, expand=True)

    # Calculate dimensions for the largest inscribed rectangle
    max_width, max_height = calculate_largest_inscribed_rectangle(original_width, original_height, angle)

    # Ensure the max dimensions do not exceed the rotated image size
    max_width = min(max_width, rotated_image.width)
    max_height = min(max_height, rotated_image.height)

    # Center coordinates of the new image
    center_x, center_y = rotated_image.width // 2, rotated_image.height // 2

    # Calculate the coordinates to crop the largest fitting rectangle
    left = max(center_x - max_width // 2, 0)
    top = max(center_y - max_height // 2, 0)
    right = min(center_x + max_width // 2, rotated_image.width)
    bottom = min(center_y + max_height // 2, rotated_image.height)

    # Crop the rotated image to the largest inscribed rectangle
    cropped_image = rotated_image.crop((left, top, right, bottom))
    return cropped_image

    
def calculate_largest_inscribed_rectangle(original_width, original_height, angle):
    # Convert the angle to radians
    angle_rad = math.radians(angle)
    
    # Calculate dimensions of the largest inscribed rectangle
    abs_cos = abs(math.cos(angle_rad))
    abs_sin = abs(math.sin(angle_rad))

    # Calculate the largest width and height of the inscribed rectangle
    largest_width = int(original_width * abs_cos + original_height * abs_sin)
    largest_height = int(original_height * abs_cos + original_width * abs_sin)

    return largest_width, largest_height


def random_augment(image):
    image_width, image_height = image.size
    # Define a list of possible transformations
    augmentations = [
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip horizontally
        transforms.RandomVerticalFlip(p=0.5),   # Randomly flip vertically
        transforms.RandomRotation(degrees=30),   # Random rotation within ±30 degrees
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),  # Random color jitter
        transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),  # Random affine transform
        transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),  # Random Gaussian blur
        transforms.RandomAdjustSharpness(sharpness_factor=2, p=0.5),  # Random sharpness adjustment
        transforms.RandomGrayscale(p=0.2),  # Randomly convert image to grayscale
        random_rotate_image
    ]
    
    # Randomly select a subset of transformations to apply
    num_transforms_to_apply = 1 #random.randint(1, len(augmentations))
    selected_transforms = random.sample(augmentations, num_transforms_to_apply)

    # Compose the selected transformations
    transform = transforms.Compose(selected_transforms)

    # Apply the transformations to the image
    augmented_image = transform(image)

    return augmented_image

def no_change(box):
    return box

def flip_boxes_horizontally(boxes, image_width):
    flipped_boxes = boxes.clone()
    flipped_boxes[:, [0, 2]] = image_width - boxes[:, [2, 0]]
    return flipped_boxes

def flip_boxes_vertically(boxes, image_height):
    flipped_boxes = boxes.clone()
    flipped_boxes[:, [1, 3]] = image_height - boxes[:, [3, 1]]
    return flipped_boxes

def rotate_boxes(boxes, angle, image_size):
    # Assumes boxes are in (xmin, ymin, xmax, ymax) format
    angle = -angle  # Reverse the angle for proper rotation
    image_center = torch.tensor(image_size[::-1]) / 2
    rot_matrix = torch.tensor([[torch.cos(angle), -torch.sin(angle)], 
                               [torch.sin(angle), torch.cos(angle)]])

    rotated_boxes = boxes.clone()

    for i, box in enumerate(boxes):
        # Get box corners
        corners = torch.tensor([[box[0], box[1]], [box[2], box[1]], [box[2], box[3]], [box[0], box[3]]])
        # Translate corners to origin
        corners -= image_center
        # Rotate corners
        rotated_corners = torch.matmul(corners, rot_matrix)
        # Translate corners back
        rotated_corners += image_center
        # Update box
        x_coords, y_coords = rotated_corners[:, 0], rotated_corners[:, 1]
        rotated_boxes[i, 0] = torch.min(x_coords)
        rotated_boxes[i, 1] = torch.min(y_coords)
        rotated_boxes[i, 2] = torch.max(x_coords)
        rotated_boxes[i, 3] = torch.max(y_coords)

    return rotated_boxes

def random_rotate_image(image):
    angle = random.uniform(-30, 30)  # Random rotation within ±30 degrees
    return F.rotate(image, angle)


def get_activation(activation, img):
    w,h = img.size

    # Normalize the Grad-CAM
    grad_cam = torch.nn.functional.relu(activation)
    grad_cam = torch.nn.functional.interpolate(grad_cam.unsqueeze(0), size=(w,h), mode='bilinear', align_corners=False)
    grad_cam = grad_cam.detach().squeeze().cpu().numpy()
    grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())
    return grad_cam


def disp_activation(activation, img, ax = None, title = None):

    if ax is None:
        _, ax = plt.subplots(1)

    grad_cam = get_activation(activation, img)

    # Display original image with heatmap
  
    heatmap_img = np.uint8(255 * grad_cam)
    heatmap_img = Image.fromarray(heatmap_img).resize(img.size)
    heatmap_img = np.array(heatmap_img)
    heatmap_img = plt.cm.jet(heatmap_img)[:, :, :3]  # Convert to RGB

    # Combine heatmap with original image
    overlayed_img = np.float32(img) / 255 + np.float32(heatmap_img)
    overlayed_img = overlayed_img / overlayed_img.max()

    ax.imshow(overlayed_img)
    if title is not None:
        ax.set_title(title)