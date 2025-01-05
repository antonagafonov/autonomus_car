import cv2
import numpy as np
import ast  # To safely convert string to dictionary
import tqdm  # For progress bar

# Path to the data and images
folder_name = 'data_sony'
data_path = 'F:/AA-Private/car/all_data/'+folder_name+'/data.txt'
images_path = 'F:/AA-Private/car/all_data/'+folder_name+'/'

# Read the data.txt file
with open(data_path, 'r') as data_file:
    lines = data_file.readlines()

# Create a video writer
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter('F:/AA-Private/car/all_data/'+folder_name+'/video_bottom.avi', fourcc, 8.0, (640, 480))

# Iterate over all the lines in the data.txt file
for idx, line in tqdm.tqdm(enumerate(lines)):
    # Parse the line as a dictionary
    try:
        state = ast.literal_eval(line.strip())  # Safely convert string to dictionary
    except (ValueError, SyntaxError) as e:
        print(f"Error parsing line {idx}: {line} - {e}")
        continue
    
    # Extract steering and throttle values
    steering = state.get('steering', 0)
    throttle = state.get('forward', 0)  # Assuming throttle corresponds to 'forward'
    # 28:11:2024:07:28:37:704
    timestep = state.get('timestep', 0)
    timestep = timestep[11:]

    # Get the image path
    img_path = f'{images_path}{idx:06d}.png'

    # Read the image
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (640, 480))  # Resize the image to 640x480
    # convert BGR to RGB
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # Cut the bottom part of the image (240 pixels)
    bot = img[240:, :, :]  # Crop from 240th pixel to the bottom

    # Make a zero array with the same size as the top part (240x640)
    mid = np.zeros_like(img)

    # Place the cropped bottom part (`bot`) in the middle of `top`
    mid[120:360, :, :] = bot  # Adjust this placement if needed

    if img is None:
        print(f"Error reading image {img_path}")
        continue

    # Put text on the image in the right corner
    cv2.putText(mid, f'Steering: {steering}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)  # Green
    cv2.putText(mid, f'Throttle: {throttle}', (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)  # Blue
    cv2.putText(mid, f'Timestep: {timestep}', (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)  # Red

    # Add the image to the video
    out.write(mid)

# Release the video writer
out.release()
print('Video saved')
