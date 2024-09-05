import torch
import random
from torchvision import transforms
import torchvision.transforms.functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from torchvision.transforms.functional import to_pil_image
from PIL import Image
import torchvision.transforms.functional as F
import torch
import cv2
import numpy as np

def flip_boxes_horizontally(boxes, image_width):
    flipped_boxes = boxes.clone()
    flipped_boxes[:, [0, 2]] = image_width - boxes[:, [2, 0]]
    return flipped_boxes

def flip_boxes_vertically(boxes, image_height):
    flipped_boxes = boxes.clone()
    flipped_boxes[:, [1, 3]] = image_height - boxes[:, [3, 1]]
    return flipped_boxes

def rotate_image_and_boxes(image, boxes, angle):
    """Rotate the image and adjust bounding boxes accordingly.
    Angle should be one of [0, 90, 180, 270]."""

    # Rotate the image
    rotated_image = F.rotate(image, angle, expand=True)

    # Calculate new box coordinates
    if angle == 90:
        new_boxes = torch.stack([
            boxes[:, 1], 
            image.width - boxes[:, 2], 
            boxes[:, 3], 
            image.width - boxes[:, 0]
        ], dim=1)
    elif angle == 180:
        new_boxes = torch.stack([
            image.width - boxes[:, 2], 
            image.height - boxes[:, 3], 
            image.width - boxes[:, 0], 
            image.height - boxes[:, 1]
        ], dim=1)
    elif angle == 270:
        new_boxes = torch.stack([
            image.height - boxes[:, 3], 
            boxes[:, 0], 
            image.height - boxes[:, 1], 
            boxes[:, 2]
        ], dim=1)
    else:
        new_boxes = boxes.clone()

    # Ensure boxes are within bounds
    new_boxes[:, [0, 2]] = new_boxes[:, [0, 2]].clamp(0, rotated_image.width)
    new_boxes[:, [1, 3]] = new_boxes[:, [1, 3]].clamp(0, rotated_image.height)

    return rotated_image, new_boxes


def random_augment_ft(image, target):
    # image is a PIL image

    # Extract bounding boxes
    boxes = target['boxes'].clone()  # Clone to avoid in-place modification
    image_width, image_height = image.size
    
    # Define transformations that need manual box adjustment
    def apply_transformations(img, boxes):
        # Random Horizontal Flip
        if random.random() > 0.5:
            img = F.hflip(img)
            boxes = flip_boxes_horizontally(boxes, image_width)
        
        # Random Vertical Flip
        if random.random() > 0.5:
            img = F.vflip(img)
            boxes = flip_boxes_vertically(boxes, image_height)
        
        # Random Rotation
        if random.random() > 0.5:
            angle = random.choice([90, 180, 270])  # Removed 0 to avoid redundant rotations
            img, boxes = rotate_image_and_boxes(img, boxes, angle)  # Use transformed img, not original image
        
        # Apply other transformations that don't need box adjustment
        # img = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2)(img)
        img = transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0))(img)
        
        return img, boxes

    # Apply transformations
    augmented_image, transformed_boxes = apply_transformations(image, target['boxes'].clone())
    
    # Update target with transformed boxes
    transformed_target = target.copy()
    transformed_target['boxes'] = transformed_boxes

    return augmented_image, transformed_target

def create_aug_batch(image,target,batch_size=4):
    im_batch, tar_batch  = [], []
    for _ in range(batch_size):
        images, new_target = random_augment_ft(image.copy(), target)
        images = transforms.ToTensor()(images)
        im_batch.append(images)
        tar_batch.append(new_target)
    return im_batch, tar_batch 


def plot_image_with_boxes(image, boxes):
    # Convert the image to a format suitable for plotting
    # image = F.to_pil_image(image)  # Convert tensor to PIL image if needed

    # Create a figure and axis
    fig, ax = plt.subplots(1)

    # Display the image
    ax.imshow(image)

    # Plot each bounding box
    for box in boxes:
        xmin, ymin, xmax, ymax = box # pascal_voc
        width, height = xmax - xmin, ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)

    # Show the plot
    plt.show()


def plot_predictions(image, predictions, threshold=0.5, ax = None):

    if not isinstance(image,np.ndarray):
        if isinstance(image,torch.Tensor):
            image = image.squeeze().permute(1, 2, 0).numpy()
        else:
            image = np.array(image)

    # Create a figure and axis
    if ax is None:
        _, ax = plt.subplots(1)

    for i, box in enumerate(predictions[0]['boxes']):
        score = predictions[0]['scores'][i]
        if score >= threshold:
            x_min, y_min, x_max, y_max = box

            # Draw a rectangle around the detected object
            cv2.rectangle(image, 
                          (int(x_min), int(y_min)), 
                          (int(x_max), int(y_max)), 
                          (0, 255, 0), 2)

            # Put the label on the object
            label = int(predictions[0]['labels'][i])
            label_text = f"Class {label}: {score:.2f}"
            cv2.putText(image, label_text, 
                        (int(x_min), int(y_min)-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 2)
        else:
            break
    ax.imshow(image)
    ax.axis('off')  # Hide axes if not needed