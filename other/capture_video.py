import cv2
import time

# Initialize video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Camera not accessible")
    exit()

def getImg(display=False, size=[480, 240]):
    """Capture and resize an image frame."""
    ret, frame = cap.read()
    # img = cv2.resize(img, (size[0], size[1]))
    # if display:
    #     cv2.imshow('IMG', img)
    return ret, frame

def main():
    recording = False
    out = None
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for .avi format
    fps = 30  # Frames per second
    output_filename = None

    print("Press 'c' to start/stop recording, 'q' to quit.")
    
    while True:
        ret,img = getImg(True)  # Display the current frame
        if not ret:
            print("Error: Couldn't capture frame")
            break
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('c'):
            if not recording:
                # Start recording
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                output_filename = f"recording_{timestamp}.avi"
                out = cv2.VideoWriter(output_filename, fourcc, fps, (img.shape[1], img.shape[0]))
                print(f"Started recording: {output_filename}")
                recording = True
            else:
                # Stop recording
                print(f"Stopped recording: {output_filename}")
                recording = False
                if out:
                    out.release()
                    out = None

        elif key == ord('q'):
            # Quit program
            print("Exiting the program...")
            break

        # Write frame to video file if recording
        if recording and out:
            out.write(img)
    
    # Release resources
    cap.release()
    if out:
        out.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
