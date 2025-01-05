import pygame
from pygame.locals import *
import time

def initialize_controller():
    pygame.init()
    pygame.joystick.init()

    # Check for connected joysticks
    joystick_count = pygame.joystick.get_count()
    print(f"Detected {joystick_count} joystick(s)")
    if joystick_count == 0:
        print("No joystick detected!")
        return None
    
    # Initialize the first joystick
    joystick = pygame.joystick.Joystick(0)
    # # Initialize the second joystick
    # joystick = pygame.joystick.Joystick(1)
    
    joystick.init()
    print(f"Joystick detected: {joystick.get_name()}")
    return joystick

def print_controller_input(joystick):
    """Poll controller inputs and print button press and release events."""
    try:
        while True:
            for event in pygame.event.get():  # Process all events
                if event.type == pygame.JOYBUTTONDOWN:
                    print(f"Button {event.button} pressed")
                elif event.type == pygame.JOYBUTTONUP:
                    print(f"Button {event.button} released")

                # Handle axes (analog sticks)
                if event.type == pygame.JOYAXISMOTION:
                    axis_value = joystick.get_axis(event.axis)
                    if abs(axis_value) > 0.1:  # Threshold to ignore small movements
                        print(f"Axis {event.axis} moved to {axis_value:.2f}")

                # Handle D-pad (hats)
                if event.type == pygame.JOYHATMOTION:
                    hat_value = joystick.get_hat(event.hat)
                    if hat_value != (0, 0):  # Ignore neutral state
                        print(f"Hat {event.hat} moved to {hat_value}")

            time.sleep(0.1)  # Limit polling rate
    except KeyboardInterrupt:
        print("\nExiting...")
        pygame.quit()

if __name__ == "__main__":
    joystick = initialize_controller()
    if joystick:
        print_controller_input(joystick)
