import torch
from .utils import random_rotate_image, random_augment
from .loss_func import nt_xent_loss

# Pre-training function
def pre_train_one_image(model, image, optimizer = None, scheduler= None, loss_fn = None, epochs=10,
                         patience=3, save_path='best_model.pth', update_callback=None):
    
    if optimizer is None:
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if scheduler is None:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

    if loss_fn is None:
        loss_fn = nt_xent_loss


    best_loss = float('inf')
    best_epoch = 0
    patience_counter = 0

    model.train()
    
    for epoch in range(epochs):
        new_im = random_augment(image)
        optimizer.zero_grad()

        # Forward pass
        zi, zj, za = model(new_im)
        loss = loss_fn(zi, zj, za)

        # Backward pass
        loss.backward()
        optimizer.step()

        # Step the scheduler with the current loss
        scheduler.step(loss.item())

        print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}')

        # Update the loss in the GUI
        if update_callback is not None:
            update_callback(epoch, loss.item())
            
        # Check if the current loss is the best we've seen so far
        if loss.item() < best_loss:
            best_loss = loss.item()
            best_epoch = epoch
            patience_counter = 0

            # Save the best model
            torch.save(model.state_dict(), save_path)
            print(f"Best model saved with loss {best_loss} at epoch {best_epoch + 1}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= patience:
            print(f"Early stopping triggered after {patience} epochs with no improvement.")
            break

    # Optionally load the best model after training
    model.load_state_dict(torch.load(save_path))
    print(f"Training completed. Best model from epoch {best_epoch + 1} loaded.")

# # Define the necessary components
# loss_fn = nt_xent_loss
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=2, verbose=True)

# Run the pre-training loop
# pre_train_one_image(model, im, optimizer, scheduler, loss_fn, epochs=10, patience=3, save_path='best_model.pth')
