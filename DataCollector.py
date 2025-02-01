import threading
import time
import os
import cv2
import numpy as np
import copy
import queue

class DataCollector(threading.Thread):
    def __init__(self, output_dir="data"):
        """Initialize the data collector thread."""
        super().__init__()
        self.output_dir = output_dir
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            print(f"Output directory created: {self.output_dir}")
        
        self.data = []
        self.running = True
        self.idx = 0
        self.task_queue = queue.Queue()  # Task queue for saveData

        print("Data collector initialized.")

    def run(self):
        """Thread run method to process queued tasks."""
        while self.running or not self.task_queue.empty():
            try:
                task = self.task_queue.get(timeout=1)  # Get task from queue
                if task:
                    image, state, idx = task
                    self._save_data_internal(image, state, idx)
            except queue.Empty:
                pass  # Continue looping if queue is empty

    def saveData(self, image, state):
        """Queue the image and state for saving asynchronously."""
        self.task_queue.put((image, state, self.idx))
        self.idx += 1

    def _save_data_internal(self, image, state, idx):
        """Save the image and state internally."""
        name = f"{idx:06d}.png"
        filename = os.path.join(self.output_dir, name)
        cv2.imwrite(filename, cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        save_state = copy.deepcopy(state)
        save_state["idx"] = idx
        save_state["image"] = name
        self.data.append(save_state)

    def stop(self):
        """Stop the data collector thread."""
        self.running = False
        self.join()  # Wait for the thread to finish processing

    def save_data_to_file(self):
        """Save the collected data to a file."""
        filename = os.path.join(self.output_dir, "data.txt")
        with open(filename, "w") as f:
            for state in self.data:
                f.write(f"{state}\n")

if __name__ == "__main__":
    data_collector = DataCollector()
    data_collector.start()  # Start the thread

    try:
        for i in range(5):
            print("Collecting data...")
            state = {'steering': -1, 'forward': 1, 'backward': 4, 'boost': 5, 'exit': 6}
            image = np.zeros((480, 640, 3), dtype=np.uint8)
            data_collector.saveData(image, state)  # Asynchronous call
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping the data collector...")
    finally:
        data_collector.save_data_to_file()
        data_collector.stop()
        print("Data collector stopped.")
