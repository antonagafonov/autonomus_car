import cv2
import os

def capture_image():
    # Open the camera (0 for default camera)
    cap = cv2.VideoCapture(1)

    # Check if the camera is opened correctly
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Capture a single frame
    ret, img = cap.read()
    img = cv2.resize(img,(480,240))
        # Check if the captured image is black
    if cv2.countNonZero(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)) == 0:
        print("Error: Captured image is black.")
        return
    # Check if the frame is captured successfully
    if not ret:
        print("Error: Failed to capture image.")
        return

    # Save the captured image in the same directory as the script
    image_path = os.path.join(os.getcwd(), "captured_image.jpg")
    cv2.imwrite(image_path, img)
    print(f"Image saved at: {image_path}")

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capture_image()
