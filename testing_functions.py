import cv2
import torch
import torchvision.transforms as T
# from torchvision.models.detection import retinanet_resnet50_fpn
import matplotlib.pyplot as plt

def livecam_test(model):

    # Initialize the RetinaNet model
    # model = retinanet_resnet50_fpn(pretrained=True)
    # model.eval()

    # Initialize video capture from the camera
    # cap = cv2.VideoCapture(0)  # 0 is the default camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Set width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Set height
    # Define the transformation
    transform = T.Compose([
        T.ToTensor(),
    ])

    # Function to plot the predictions on the frame
    def plot_predictions(frame, predictions, threshold=0.5):
        for i, box in enumerate(predictions[0]['boxes']):
            score = predictions[0]['scores'][i]
            if score >= threshold:
                x_min, y_min, x_max, y_max = box

                # Draw a rectangle around the detected object
                cv2.rectangle(frame, 
                            (int(x_min), int(y_min)), 
                            (int(x_max), int(y_max)), 
                            (0, 255, 0), 2)

                # Put the label on the object
                label = int(predictions[0]['labels'][i])
                label_text = f"Class {label}: {score:.2f}"
                cv2.putText(frame, label_text, 
                            (int(x_min), int(y_min)-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            0.5, (0, 255, 0), 2)

    # Start capturing video
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to a tensor and add a batch dimension
        input_tensor = transform(frame).unsqueeze(0)

        # Make predictions
        with torch.no_grad():
            predictions = model.P2.retinanet(input_tensor)

        # Plot predictions on the frame
        plot_predictions(frame, predictions)

        # Display the frame
        cv2.imshow('RetinaNet Live Feed', frame)

        # Break loop on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            cap.release()
            break

    # Release the video capture object and close display windows
    cv2.destroyAllWindows()
    cv2.waitKey(1) # MacOS bs...