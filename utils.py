import threading
import RPi.GPIO as GPIO
from time import sleep
import lgpio
import numpy as np
from datetime import datetime
import cv2
import matplotlib.pyplot as plt

# Function to save image in a separate thread
def save_images(im_dict):
    # read the images from dictionary and save
    for key in im_dict:
        cv2.imwrite(key, im_dict[key])

def close_open(im,ksize = 21):
    # Define a kernel for morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))  # Adjust size as needed

    # Perform morphological closing to fill small holes in the lane
    im = cv2.morphologyEx(im, cv2.MORPH_CLOSE, kernel)

    # Perform morphological opening to remove noise while preserving the lane structure
    im = cv2.morphologyEx(im, cv2.MORPH_OPEN, kernel)
    return im

def segment_green_lanes(image):
    """
    Segment green lanes from the input image.
    
    Parameters:
    - image: Input image (BGR format).
    
    Returns:
    - mask: Binary mask highlighting green lanes.
    - result: Resulting image with lanes highlighted.
    """
    # Convert image to HSV color space
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define lower and upper bounds for green color in HSV
    lower_green = np.array([35, 40, 40])  # Adjust based on your green hue
    upper_green = np.array([85, 255, 255])

    # Threshold the HSV image to get only green colors
    mask = cv2.inRange(hsv, lower_green, upper_green)

    # Apply morphological operations to clean up the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close small gaps
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove small noise

    # Optional: Apply Gaussian blur to smooth the mask
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    # Use the mask to segment the green areas
    result = cv2.bitwise_and(image, image, mask=mask)

    return mask, result

def lane_from_R_inv(image,trsh = 180):
    # make inverse of red channel
    R = 255 - image[:, :, 0]
    # apply gaussian blur
    R = cv2.GaussianBlur(R, (21, 21), 0)
    # apply median blur
    R = cv2.medianBlur(R, 21)
    # apply threshold
    R_trsh = cv2.threshold(R, trsh, 255, cv2.THRESH_BINARY)[1]
    return R_trsh

def white_balance(img):
    """
    Apply white balance correction using the Gray World Assumption.
    
    Args:
        img (numpy.ndarray): Input BGR image.
    
    Returns:
        numpy.ndarray: White-balanced BGR image.
    """
    result = img.copy()
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])

    avg_gray = (avg_b + avg_g + avg_r) / 3

    scale_b = avg_gray / avg_b
    scale_g = avg_gray / avg_g
    scale_r = avg_gray / avg_r

    result[:, :, 0] = np.clip(result[:, :, 0] * scale_b, 0, 255)
    result[:, :, 1] = np.clip(result[:, :, 1] * scale_g, 0, 255)
    result[:, :, 2] = np.clip(result[:, :, 2] * scale_r, 0, 255)

    return result.astype(np.uint8)


def hsv_green_mask(image,trsh = 120):
    image = white_balance(image)

    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # # Define a broader green color range
    # lower_green = np.array([120, 80, 80])  # Adjusted lower bound for green in HSV
    # upper_green = np.array([180, 180, 180])  # Adjusted upper bound for green in HSV

    # Define a broader green color range
    lower_green = np.array([20, 60, 20])  # Adjust if needed
    upper_green = np.array([80, 255, 255]) 
    
    # Create a mask to detect green
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # apply gaussian blur
    mask = cv2.GaussianBlur(mask, (21, 21), 0)
    # apply median blur
    mask = cv2.medianBlur(mask, 21)
    # apply threshold on mask
    mask = cv2.threshold(mask, trsh, 255, cv2.THRESH_BINARY)[1]

    return mask

def process_image(im_input,cut_threshold = 190,thresh_percentile = 95):
    # img = im_input[200:,:,:]
    img = im_input[200:440,:,:]
    width, height = img.shape[1], img.shape[0]
    # convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    threshold = hsv_green_mask(img)

    # img_green_channel = img_rgb[:,:,1]

    # # convert to grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # # apply gaussian blur
    img_blured = cv2.GaussianBlur(img_gray, (21, 21), 0)
    # img_green_channel_blured = cv2.GaussianBlur(img_green_channel, (21, 21), 0)

    # # invert the image
    inversed = cv2.bitwise_not(img_blured)
    
    # find mean and 0.3 top bright colors threshold of the image for thresholding
    # mean = np.mean(inversed)
    # dynamic_cut_threshold = int(mean * 1.3)
    # print("cut_threshold: ",dynamic_cut_threshold)
    # threshold the image 140 to 255
    
    # dynamic_thresh = np.percentile(inversed, thresh_percentile)
    
    # print("dynamic_thresh: ",dynamic_thresh)
    # night_tresh = 220
    # torch_tresh = 170

    # _, threshold = cv2.threshold(cv2.GaussianBlur(cv2.bitwise_not(img_rgb[:,:,2]), (21, 21), 0), cut_threshold, 255, cv2.THRESH_BINARY) # night threshold
    
    # _, threshold = cv2.threshold(inversed, cut_threshold, 255, cv2.THRESH_BINARY) # day threshold
    # print("threshold: ",threshold.shape)
    # make triangle cut on the top right corner and on the top left corner from middle of the heeight to middle of the width
    # fill with zeros top part of threshold
    threshold[:height//2, :] = 0
    cutted_threshold = threshold.copy()

    # cutted_threshold[:height//2, width//2:] = 0
    # cutted_threshold[:height//2, :width//2] = 0
    # # Define the vertices for the left triangle (bottom-left to top-left quarter width)
    # left_triangle = np.array([
    #     [0, height],         # Bottom-left corner
    #     [width // 3, 0],     # Top-left quarter of the width
    #     [0, 0]               # Top-left corner
    # ], dtype=np.int32)

    # # Define the vertices for the right triangle (bottom-right to top-right three-quarters width)
    # right_triangle = np.array([
    #     [width, height],        # Bottom-right corner
    #     [width, 0],             # Top-right corner
    #     [3 * width // 3, 0]     # Top-right three-quarters of the width
    # ], dtype=np.int32)

    # # Create masks for the triangles
    # cv2.fillPoly(cutted_threshold, [left_triangle], 0)  # Mask out left triangle
    # cv2.fillPoly(cutted_threshold, [right_triangle], 0)  # Mask out right triangle
    # # apply median blur on cutted_threshold
    # cutted_threshold = cv2.medianBlur(cutted_threshold, 5)
    
    # cv2.imwrite(f"/home/toon/data/blue_cutted_trsh_temp.png", cutted_threshold)
    # cutted_threshold = close_open(cutted_threshold,ksize = 21)

    countours_img = img_rgb.copy()
    return inversed,cutted_threshold, img_rgb,countours_img,width,height

def process_contours(cutted_threshold,xl=20,xr=620,min_area=500,max_area = 6000):

    # find all clusters of white pixels in cutted_threshold
    contours, _ = cv2.findContours(cutted_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    filtered_contours = []

    for contour in contours:
        # print("len(contour):",len(contour),"cv2.contourArea(contour):",cv2.contourArea(contour))
        M = cv2.moments(contour)
        if M["m00"] != 0:  # check if contour is non-zero area
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            # print(cX, cY)
            # Filter contours based on x-coordinate of centroid
            if (xl < cX < xr) and (cv2.contourArea(contour) > min_area) and (cv2.contourArea(contour) < max_area):
                filtered_contours.append(contour)

    return filtered_contours

def print_countour_info(contours,countours_img):
    print('Contours:', len(contours))
    for contour in contours:
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(countours_img, (cX, cY), 5, (0, 255, 0), -1)
            print(cX, cY)
    return countours_img

def create_line(cX = None, cY = None, contour = None,side = None, countours_img = None):
    # Reshape the contour to extract points
    contour_points = contour.reshape(-1, 2)

    # Sort the points by Y-coordinate (index 1) to find the top and bottom points
    top_points = contour_points[contour_points[:, 1].argsort()][:5]

    # Now, find the top-left, top-right, and bottom points
    top_left = top_points[np.argmin(top_points[:, 0])]
    top_right = top_points[np.argmax(top_points[:, 0])]

    # Draw the top-left, top-right, bottom-left, and bottom-right points
    cv2.circle(countours_img, (top_left[0], top_left[1]), 5, (0, 0, 255), -1)
    cv2.circle(countours_img, (top_right[0], top_right[1]), 5, (0, 0, 255), -1)

    # Draw lines from the center to the four corner points
    if side == 'right':
        cv2.line(countours_img, (cX, cY), tuple(top_right), (0, 255, 0), 2)
        return (cX, cY) , tuple(top_right)
    elif side == 'left':
        cv2.line(countours_img, (cX, cY), tuple(top_left), (0, 255, 0), 2)
        return (cX, cY) , tuple(top_left)
    else:
        print('Invalid side parameter. Please use "right" or "left".')
        return None
    
def calculate_slope(contours = None):
    # get all points inside the contour
    points = contours.reshape(-1, 2)
    # now find top mid poin and bottom mid point
    top_points = points[points[:, 1].argsort()][:5]
    top_left = top_points[np.argmin(top_points[:, 0])]
    top_right = top_points[np.argmax(top_points[:, 0])]
    top_mid = (top_left + top_right) // 2
    # find bottom mid point
    bottom_points = points[points[:, 1].argsort()][-5:]
    bottom_left = bottom_points[np.argmin(bottom_points[:, 0])]
    bottom_right = bottom_points[np.argmax(bottom_points[:, 0])]
    bottom_mid = (bottom_left + bottom_right) // 2
    # calculate slope
    slope = (bottom_mid[1] - top_mid[1]) / (bottom_mid[0] - top_mid[0])
    return slope

def get_motor_action(   filtered_contours=None,
                        width=None,
                        img= None,
                        height=None,
                        one_line_delta = 35,
                        right_turn_coeff = 1.7
                    ):
    steer_action, speed_action = None, None
    angle_r, angle_l = None, None
    deviation = None

    # if no contours detected, stop the motors
    if len(filtered_contours) == 0:
        # print('No contours detected, stopping motors.')
        steer_action, speed_action,deviation = None, None, None

    # one countour detected
    elif len(filtered_contours) == 1:
        # print('One contour detected, steering outwards this line.')
        M = cv2.moments(filtered_contours[0])
        # getting the centroid of the contour
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

        slope = calculate_slope(filtered_contours[0])
        # print('Slope:', slope)
        if slope < -0.1:
        # if cX < width // 2 - (width // 4):
            # steer right
            steer_action = 1
            deviation = int(one_line_delta*right_turn_coeff)
        elif slope > 0.1:
        # elif cX > width // 2 + (width // 4):
            # steer left
            steer_action = -1
            deviation = -one_line_delta
        else:
            # steer straight
            # print('Contour is in the center, steering straight.')
            steer_action, speed_action,deviation = 0, 1, 0
    # two countours detected
    elif len(filtered_contours) == 2:
        # print('Two contours detected, steering inwards.')
        # check if centers of both lanes are in the middle of each contour, if yes, go straight
        M1 = cv2.moments(filtered_contours[0])
        M2 = cv2.moments(filtered_contours[1])
        # getting the centroid of the contours
        if M1["m00"] != 0:
            cX1 = int(M1["m10"] / M1["m00"])
            cY1 = int(M1["m01"] / M1["m00"])
        if M2["m00"] != 0:
            cX2 = int(M2["m10"] / M2["m00"])
            cY2 = int(M2["m01"] / M2["m00"])

        if cX1 > cX2:
            # cX1 is on the right and we work with it
            cX_r, cY_r = cX1, cY1
            contour_r = filtered_contours[0]
            cX_l, cY_l = cX2, cY2
            contour_l = filtered_contours[1]
        else:
            # cX2 is on the right and we work with it
            cX_r, cY_r = cX2, cY2
            contour_r = filtered_contours[1]
            cX_l, cY_l = cX1, cY1
            contour_l = filtered_contours[0]

        # find right line using create_line function
        _, top_right = create_line(cX_r, cY_r, contour_r, side='left', countours_img=img)
        # find left line using create_line function
        _, top_left = create_line(cX_l, cY_l, contour_l, side='right', countours_img=img)
        # draw both lines with different collor on img
        # draw center line which is between top_right and top_left and vertically down to the bottom of the image
        midpoint_x, midpoint_y = (top_right[0] + top_left[0]) // 2, (top_right[1] + top_left[1]) // 2

        # calculate deviation from center 
        deviation = midpoint_x - width // 2
        # draw center line
        cv2.line(img, (width//2, midpoint_y), (width//2, height), (255, 0, 0), 6)
        # draw in between lines line
        cv2.line(img, (midpoint_x, midpoint_y), (midpoint_x, height), (0, 255, 255), 2)

        cv2.line(img, (cX_r, cY_r), top_right, (0, 255, 0), 2)
        cv2.line(img, (cX_l, cY_l), top_left, (0, 255, 0), 2)
        # calculate angle of the right line and left line in degrees
        angle_r = np.arctan2(top_right[1] - cY_r, top_right[0] - cX_r) * 180 / np.pi
        angle_l = np.arctan2(top_left[1] - cY_l, top_left[0] - cX_l) * 180 / np.pi     
        # add 360 to angle if it is negative
        if angle_r < 0:
            angle_r += 360
        if angle_l < 0:
            angle_l += 360
        # now we deside what we have straight, right turn steer or left turn steer
        if angle_l - angle_r > 70:
            # straight
            steer_action, speed_action = 0, 1
        elif angle_l - angle_r < 50 and angle_r < 230 :
            # steer left proportionaly to 45/(angle_l - angle_r) 
            steer_action, speed_action = -1 * 45/(angle_l - angle_r), 1
        elif angle_l - angle_r < 50 and angle_r > 260 :
            # steer right proportionaly to 45/(angle_r - angle_l) 
            steer_action, speed_action = 1 * 45/(angle_r - angle_l), 1

    return len(filtered_contours),steer_action, speed_action, angle_r, angle_l,deviation, img

class PIDController:
    def __init__(self):
        """
        Initialize PID controller with given gains.

        Parameters:
        - Kp: Proportional gain
        - Ki: Integral gain
        - Kd: Derivative gain
        """

        self.Kp_l = 0.008
        self.Ki_l = 0.001
        self.Kd_l = 0.0002

        # PID parameters for positive deviation
        self.Kp_r = 0.008 
        self.Ki_r = 0.001
        self.Kd_r = 0.0002

        self.prev_error = 0.0
        self.integral = 0.0

    def control(self, cte, dt):
        """
        Compute the steering angle using PID control.

        Parameters:
        - cte: Cross-track error
        - dt: Time difference

        Returns:
        - Steering angle adjustment
        
        """

        # Use different PID parameters based on the sign of the deviation (CTE)
        if cte > 0.05:
            # Positive deviation, use positive PID parameters
            Kp = self.Kp_r
            Ki = self.Ki_r
            Kd = self.Kd_r
        else:
            # Negative deviation, use regular PID parameters
            Kp = self.Kp_l
            Ki = self.Ki_l
            Kd = self.Kd_l


        # Proportional term
        P = Kp * cte

        # Integral term
        self.integral += cte * dt
        I = Ki * self.integral

        # Derivative term
        derivative = (cte - self.prev_error) / dt if dt > 0 else 0.0
        D = Kd * derivative

        # Update error
        self.prev_error = cte

        # Return control output
        return round((P + I + D),4)  # Negative sign for correction
    def reset(self):
        self.prev_error = 0.0
        self.integral = 0.0
        
def get_deviation(im_input):
    pid_steer, pid_speed = None, None

    inversed,cutted_threshold, img_rgb,countours_img,width,height = process_image(im_input)
    
    contours = process_contours(cutted_threshold)

    cv2.drawContours(countours_img, contours, -1, (255, 0, 0), 3)

    countours_img = print_countour_info(contours,countours_img)

    len_contours,_ , _, _, _,deviation,countours_img = get_motor_action(    filtered_contours = contours,
                                                                            width=cutted_threshold.shape[1],
                                                                            img = countours_img,
                                                                            height=cutted_threshold.shape[0])   
    print("deviation: ",deviation)
    return len_contours,deviation,countours_img,cutted_threshold,inversed

def get_time():
    now = datetime.now()
    formatted_time = now.strftime("%d:%m:%Y:%H:%M:%S") + f":{now.microsecond // 1000:02d}"
    return formatted_time

def calibrate_steering(turn,max_turn = 0.8):
    if turn > 0:
        return max(turn-(1-max_turn),0)
    elif turn < 0:
        return min(turn+(1-max_turn),0)
    else:
        return 0
    
def get_time():
    now = datetime.now()
    formatted_time = now.strftime("%d:%m:%Y:%H:%M:%S") + f":{now.microsecond // 1000:02d}"
    return formatted_time

def create_control_signal(prediction, idx, record = False):
    if record:
        r = 1
    else:
        r = 0
    curr = [0.0,0.0]
    if idx>=10: # if we have reached the end of the prediction, stop the vehicle
        fw = 0.0
        bw = 0.0
        curr[0] = 0.0
        
    else:
        # Create a control signal based on the prediction
        curr = prediction[:, idx]
        print("curr: ",curr)
        if curr[1]>0.2:
            fw = curr[1]
            bw = 0.0
        elif curr[1]<-0.2:
            fw = 0.0
            bw = curr[1]
        else:
            fw = 0.0
            bw = 0.0

    state = {
    "steering": round(float(curr[0]),4),  # -1 (left), 0 (center), 1 (right)
    "forward": round(float(fw),4),   # 0 (off), 1 (on)
    "backward": bw,
    "boost": 0,   # 0 (off), 1 (on)
    "recording": 1,  # 0 (off), 1 (on)
    "exit": 0 ,      # 0 (off), 1 (on)
    "timestep": get_time()
            }
    
    return state, idx+1

class VehicleSteering(threading.Thread):
    def __init__(self, input_queue, stop_event):
        """
        Initialize the VehicleSteering thread.
        :param input_queue: A queue to receive input commands.
        :param stop_event: An event to signal stopping the thread.
        """
        super().__init__()
        self.input_queue = input_queue
        self.stop_event = stop_event

        self.cleanup_done = False  # Add a flag to track cleanup
        self.cleanup_lock = threading.Lock()  # Lock to synchronize cleanup

        # Motor Pins
        self.in1a = 24
        self.in2a = 23
        self.en_a = 25
        self.in1b = 20
        self.in2b = 16
        self.en_b = 21

        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setup([self.in1a, self.in2a, self.en_a, self.in1b, self.in2b, self.en_b], GPIO.OUT)
        GPIO.output([self.in1a, self.in2a, self.in1b, self.in2b], GPIO.LOW)

        # Initialize PWM
        self.pwm_a = GPIO.PWM(self.en_a, 100)
        self.pwm_b = GPIO.PWM(self.en_b, 100)
        self.pwm_a.start(0)
        self.pwm_b.start(0)

        self.startup = True
        self.mySpeed = 0
        self.straight_count = 0

    def run(self):
        """Main loop to handle motor commands."""
        while not self.stop_event.is_set():
            try:
                key = self.input_queue.get(timeout=0.05)

                if self.startup:
                    self.stop_motors()
                    self.startup = False

            except Exception:
                pass

    def move(self,speed=0.5,turn=0,boost = 0,t=0.05, steering_offset=0.0,s = 80):
        if abs(turn) < 0.1:
            self.straight_count += 1
        else:
            self.straight_count = 0

        # Apply steering offset
        if speed > 0.05:
            if abs(turn) < 0.1:
                turn += steering_offset
        speed = round(speed * (s + (100-s) * boost),1)  # Boost increases speed by 20%
        turn = round(turn * 100,1)

        # print("moving with speed: ",speed," turn: ",turn)
        leftSpeed = speed-turn
        rightSpeed = speed+turn

        if leftSpeed>100:
            leftSpeed =100

        elif leftSpeed<-100:
            leftSpeed = -100

        if rightSpeed>100:
            rightSpeed =100

        elif rightSpeed<-100:
            rightSpeed = -100

        #print(leftSpeed,rightSpeed)
        self.pwm_a.ChangeDutyCycle(abs(leftSpeed))
        self.pwm_b.ChangeDutyCycle(abs(rightSpeed))

        if leftSpeed>0:
            GPIO.output(self.in1a,GPIO.HIGH)
            GPIO.output(self.in2a,GPIO.LOW)
        elif leftSpeed < 0:
            GPIO.output(self.in1a,GPIO.LOW)
            GPIO.output(self.in2a,GPIO.HIGH)
        else:
            GPIO.output(self.in1a, GPIO.LOW)
            GPIO.output(self.in2a, GPIO.LOW)  # Stop if speed is zero

        if rightSpeed>0:
            GPIO.output(self.in1b,GPIO.HIGH)
            GPIO.output(self.in2b,GPIO.LOW)
        elif rightSpeed < 0:
            GPIO.output(self.in1b,GPIO.LOW)
            GPIO.output(self.in2b,GPIO.HIGH)
        else:
            GPIO.output(self.in1b, GPIO.LOW)
            GPIO.output(self.in2b, GPIO.LOW)
        if t>0: # Stop if speed is zero
            sleep(t)

    def stop_motors(self, t=0):
        """Stop the motors."""
        self.pwm_a.ChangeDutyCycle(0)
        self.pwm_b.ChangeDutyCycle(0)
        self.mySpeed = 0
        sleep(t)

    def set_speed(self, speed):
        """Set the current speed value."""
        self.mySpeed = speed

    def stop(self):
        with self.cleanup_lock:
            if not self.cleanup_done:

                try:
                    print("Stopping PWM and cleaning up...")
                    GPIO.cleanup()  # Clean up GPIO resources
                except Exception as e:
                    print(f"cleanup error: {e}")
                self.cleanup_done = True

    def exit_program(self):
        """Exit the program and clean up."""
        self.stop()
        self.stop_event.set()
        print("Exiting...")
