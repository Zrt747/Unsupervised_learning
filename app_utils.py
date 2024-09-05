import cv2
import tkinter as tk
from tkinter import Button
from PIL import Image, ImageTk
from tkinter import Toplevel
from threading import Thread
from model.training import pre_train_one_image
from model.network import Unsupervised_Object_Detection
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

IMAGE_PATH = "image.jpg"
PRETRAIN_EPOCHS = 10

# Function to open the camera and capture a frame
def start_app():
    global cap
    global window

    cap = cv2.VideoCapture(0)

    def show_frame():
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret:
            # Convert the image from BGR (OpenCV format) to RGB (Tkinter format)
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=img)
            lmain.imgtk = imgtk
            lmain.configure(image=imgtk)
            lmain.after(10, show_frame)

    # Stop video feed and release camera when window closes
    def on_closing():
        cap.release()
        window.destroy()

    # GUI Setup with Tkinter
    window = tk.Tk()
    window.title("Camera App")

    # # Configure the grid to ensure proper layout
    # window.grid_rowconfigure(0, weight=1)  # Allow row 0 (video feed) to expand
    # window.grid_rowconfigure(1, weight=0)  # Keep row 1 (button) fixed
    # window.grid_columnconfigure(0, weight=1)  # Center elements in column 0

    # Create a Label to show the video feed
    lmain = tk.Label(window)
    lmain.grid(row=0, column=0, padx=10, pady=10, sticky="n")  # Sticky to 'n' (top)

    # Button to capture image
    capture_button = Button(window, 
                            text="Capture Image", 
                            command=capture_image, )
    
    # Position the button at the top, below the video feed
    capture_button.grid(row=1, column=0, pady=10, sticky="n")  # Sticky to 'n' (top)

    show_frame()

    window.protocol("WM_DELETE_WINDOW", on_closing)
    window.mainloop()

# Function to capture the current frame
def capture_image():
    ret, frame = cap.read()  # Capture the current frame
    if ret:
        cv2.imwrite(IMAGE_PATH, frame)  # Save the image to disk
        print(f"Image captured and saved as {IMAGE_PATH}.")

                # Close the camera window
        window.destroy()

        # Allow the user to select a bounding box
        select_bounding_box(IMAGE_PATH)


def select_bounding_box(image_path):
    # Read the image from disk
    image = cv2.imread(image_path)
    
    # Let the user select a bounding box (drag and select)
    bbox = cv2.selectROI("Select Bounding Box", image, fromCenter=False, showCrosshair=True)
    
    # Extract the region of interest (ROI) based on the bounding box
    if bbox != (0, 0, 0, 0):  # If a bounding box was selected
        x, y, w, h = bbox
        box = [x,y,x+w,y+h]
        # roi = image[int(y):int(y+h), int(x):int(x+w)]  # Crop the ROI from the image
        print(f"Bounding box selected: {bbox}")
        
        # Close the bounding box selection window
        cv2.destroyAllWindows()
        
        # Now proceed to the training window
        model = Unsupervised_Object_Detection()
        image = Image.open(IMAGE_PATH)
        open_training_window(model, image, box)
    else:
        print("No bounding box selected, returning to main window.")


# Function to open the camera and setup pre-training
def open_training_window(model, image,box):
    global loss_label
    global current_epoch, best_loss, patience_counter
    global loss_values  # To store loss values for plotting
    
    loss_values = []

    # Create a new top-level window for training
    training_window = tk.Toplevel()
    training_window.title("Training in Progress")
    
    # Create a label to display the loss
    loss_label = tk.Label(training_window, text="Loss: Starting training...")
    loss_label.pack(pady=20)

    # Create a figure for plotting
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.set_title("Training Loss Over Epochs")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_xlim([0, PRETRAIN_EPOCHS])
    
    # Embed the plot in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=training_window)
    canvas.get_tk_widget().pack(pady=20)

    def update_loss_in_gui(epoch, loss_value):
        # Ensure the loss update happens in the main Tkinter thread using after()
        training_window.after(0, loss_label.config, {"text": f"Epoch {epoch + 1}: Loss = {loss_value:.4f}"})
        
        # Update the plot with the new loss value
        loss_values.append(loss_value)
        ax.clear()
        ax.plot(range(1, len(loss_values) + 1), loss_values, label="Loss")
        ax.legend()
        canvas.draw()


    # Run the training in a separate thread to prevent GUI freezing
    training_thread = Thread(target=pre_train_one_image, args=(model, image), kwargs={"update_callback": update_loss_in_gui})
    training_thread.start()
    
    training_window.mainloop()
    training_window.after(0, training_window.destroy)  # Close the window after training
    training_window.after(0, post_training)  # Call the post-training function



def post_training():
    print("Post-training process started...")

    # At the end of the training loop (in pre_train_one_image):
    training_window.after(0, training_window.destroy)  # Close the window after training
    training_window.after(0, post_training)  # Call the post-training function

if __name__ == '__main__':
    start_app()