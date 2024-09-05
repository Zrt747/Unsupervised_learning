import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn.functional as F
import numpy as np
from PIL import Image

def plot_activation_maps(image, model):
    # Move model to evaluation mode
    model.eval()

    # Transform image to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image).unsqueeze(0)

    # Hook function to capture the activations and gradients
    activations = []
    gradients = []

    def save_activation(name):
        def hook(model, input, output):
            activations.append(output)
        return hook

    def save_gradient(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Register hooks
    hooks = []
    names = []
    for name, layer in enumerate(model.backbone.fpn.inner_blocks):
        hooks.append(layer.register_forward_hook(save_activation(f'Inner Layer {name+1}')))
        hooks.append(layer.register_backward_hook(save_gradient))
        names.append(f'Inner Layer {name+1}')

    for i, layer in enumerate(model.backbone.fpn.layer_blocks):
        hooks.append(layer.register_forward_hook(save_activation(f'Layer {i+1}')))
        hooks.append(layer.register_backward_hook(save_gradient))
        names.append(f'Layer {i+1}')

    # Forward pass to get activations
    output = model(image_tensor)
    target_class_idx = output[0]['scores'].argmax()

    # Backward pass to get gradients
    model.zero_grad()
    output[0]['scores'][target_class_idx].backward()

    # Function to display Grad-CAM
    def display_grad_cam(activation, gradient, img, ax, layer_name):
        b, k, u, v = activation.size()

        # Calculate Grad-CAM
        alpha = gradient.view(b, k, -1).mean(2)  # Take the mean of gradients
        weights = alpha.view(b, k, 1, 1)  # Reshape for broadcasting
        grad_cam = torch.sum(weights * activation, dim=1, keepdim=True)  # Weighted sum

        # Normalize the Grad-CAM
        grad_cam = F.relu(grad_cam)
        grad_cam = F.interpolate(grad_cam, size=(img.shape[1], img.shape[2]), mode='bilinear', align_corners=False)
        grad_cam = grad_cam.detach().squeeze().cpu().numpy()
        grad_cam = (grad_cam - grad_cam.min()) / (grad_cam.max() - grad_cam.min())

        # Display original image with heatmap
        img = transforms.functional.to_pil_image(img)
        heatmap_img = np.uint8(255 * grad_cam)
        heatmap_img = Image.fromarray(heatmap_img).resize(img.size)
        heatmap_img = np.array(heatmap_img)
        heatmap_img = plt.cm.jet(heatmap_img)[:, :, :3]  # Convert to RGB

        # Combine heatmap with original image
        overlayed_img = np.float32(img) / 255 + np.float32(heatmap_img)
        overlayed_img = overlayed_img / overlayed_img.max()

        ax.imshow(overlayed_img)
        ax.set_title(layer_name)
        ax.axis('off')


    # Convert input_tensor to image format for visualization
    input_image = image_tensor.squeeze(0).clone()
    input_image = (input_image - input_image.min()) / (input_image.max() - input_image.min())

    # Visualize each layer activation as Grad-CAM
    N = len(activations)
    fig, axs = plt.subplots(1,N,figsize=(15, 5))
    for i, (activation, gradient) in enumerate(zip(activations, gradients)):
        display_grad_cam(activation, gradient, input_image, axs[i], names[i])

    # Remove hooks
    for hook in hooks:
        hook.remove()