import cv2

def draw_rectangle(image):
    global x_start, y_start, drawing

    # Callback function for mouse events
    def mouse_callback(event, x, y, flags, param):
        global x_start, y_start, drawing

        if event == cv2.EVENT_LBUTTONDOWN:
            drawing = True
            x_start, y_start = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if drawing:
                temp_image = image.copy()
                cv2.rectangle(temp_image, (x_start, y_start), (x, y), (0, 255, 0), 2)
                cv2.imshow('Image', temp_image)

        elif event == cv2.EVENT_LBUTTONUP:
            drawing = False
            cv2.rectangle(image, (x_start, y_start), (x, y), (0, 255, 0), 2)
            print(f"Bounding Box: [{x_start}, {y_start}, {x}, {y}]")

    # Initialize drawing state
    drawing = False
    x_start, y_start = -1, -1

    # Create a window and bind the function to window
    cv2.namedWindow('Image')
    cv2.setMouseCallback('Image', mouse_callback)

    while True:
        cv2.imshow('Image', image)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to exit
            break

    cv2.destroyAllWindows()

# draw_rectangle(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR))