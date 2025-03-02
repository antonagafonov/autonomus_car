import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
import re

def preprocess_timestamps(df, col = 'timestamp'):
    timestamp_col = df[col]
    # Check if the timestamps last 3 characters are milliseconds, if not add to have 3 characters
    for idx, ts in enumerate(timestamp_col):
        # 01:03:2025:10:35:09:04 -> 01:03:2025:10:35:09:040
        if len(ts.split(":")[-1]) == 2:
            ts += "0"
            # write to the dataframe
            df.at[idx, col] = ts
            print("Added 0 to timestamp:", ts, idx)
    return df

def convert_timestamp(ts):
    return int(ts.replace(":", ""))  # Remove colons and convert to integer

joystick_data = pd.read_csv("/home/toon/data/steering_data.csv")
# drop boos, backward and recording collumns from joystic_data
joystick_data = joystick_data.drop(columns=["boost", "backward","enable_pid"])
image_data = pd.read_csv("/home/toon/data/image_log.csv")

joystick_data["timestamp"] = joystick_data["timestamp"].apply(convert_timestamp)
image_data["timestamp"] = image_data["timestamp"].apply(convert_timestamp)

# Find the first joystick_data entry that is >= first image timestamp
first_image_timestamp = image_data["timestamp"].min()
print("First image timestamp:", first_image_timestamp)
for idx, row in joystick_data.iterrows():
    if row["timestamp"] <= first_image_timestamp:
        first_valid_idx = idx
        break
print("First valid index:", first_valid_idx)

# Remove all rows before this index
if first_valid_idx is not None:
    joystick_data = joystick_data.iloc[first_valid_idx:].reset_index(drop=True)


print("joystic data")
print(joystick_data.head())

print("image data")
print(image_data.head())

# Create the new dataframe
combined_data = []

for _, img_row in image_data.iterrows():
    # Find the first joystick entry where timestep >= image timestamp
    joystick_match_idx = joystick_data[joystick_data["timestamp"] >= img_row["timestamp"]].index.min()

    if pd.notna(joystick_match_idx):
        # Take the next 30 joystick indices
        joystick_idxs = joystick_data.iloc[joystick_match_idx:joystick_match_idx + 30]["idx"].tolist()
    else:
        joystick_idxs = []

    # Append the results
    combined_data.append({
        "idx": img_row["index"],
        "image_idx": img_row["index"],
        "filename": img_row["filename"],
        "joystick_idxs": joystick_idxs  # Store as list
    })


# Convert to DataFrame and save
combined_df = pd.DataFrame(combined_data)
combined_df.to_csv("/home/toon/data/combined_data.csv", index=False)

print(combined_df.head())
print("Saved combined data to /home/toon/data/combined_data.csv")