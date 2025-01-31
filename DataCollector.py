import threading
import time
import os
import cv2
import numpy as np
import copy

class DataCollector(threading.Thread):
    def __init__(self, output_dir="data"):
        """Initialize the data collector thread."""
        threading.Thread.__init__(self)
        self.output_dir = output_dir
        # if the output directory does not exist, create it
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Output directory created: {self.output_dir}")
        self.data = []
        self.running = True
        self.idx = 0
        print("Data collector initialized.")

    def saveData(self, image, state):
        """Save the image and joystick state to a file."""
        # name is 000001.png, 000002.png, ...
        name = f"{self.idx:06d}.png"
        filename = os.path.join(self.output_dir, name) 
        # save as png with opencv in RGB format
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imwrite(filename, image)
        # print("Image saved")
        save_state = copy.deepcopy(state)
        # add self.idx:06d as a first value in the dictionary
        save_state["idx"] = self.idx
        save_state["image"] = name
        self.data.append(save_state)
        # print("state in datacollector",save_state)
        self.idx += 1

    def stop(self):
        """Stop the data collector thread."""
        self.running = False

    def save_data_to_file(self):
        """Save the collected data to a file."""
        filename = os.path.join(self.output_dir, "data.txt")
        with open(filename, "w") as f:
            for state in self.data:
                f.write(f"{state}\n")

if __name__ == "__main__":
    data_collector = DataCollector()
    data_collector.start()
    try:
        for i in range(5):
            print("Collecting data...")
            # Simulate the joystick state
            state = {'steering': -1, 'forward': 1, 'backward': 4, 'boost': 5, 'exit': 6}
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            data_collector.saveData(image, state)
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping the data collector...")
    finally:
        data_collector.save_data_to_file()
        data_collector.stop()
        data_collector.join()
        print("Data collector stopped.")