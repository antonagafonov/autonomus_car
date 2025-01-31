import smbus
from time import sleep
import numpy as np
import os
import sys
import matplotlib.pyplot as plt

# MPU6050 Registers and their Address
PWR_MGMT_1 = 0x6B
SMPLRT_DIV = 0x19
CONFIG = 0x1A
GYRO_CONFIG = 0x1B
INT_ENABLE = 0x38
ACCEL_XOUT_H = 0x3B
ACCEL_YOUT_H = 0x3D
ACCEL_ZOUT_H = 0x3F
GYRO_XOUT_H = 0x43
GYRO_YOUT_H = 0x45
GYRO_ZOUT_H = 0x47

# Sampling time (s)
DT = 0.1
ALPHA = 0.9  # High-pass filter coefficient

def MPU_Init():
    bus.write_byte_data(Device_Address, SMPLRT_DIV, 7)
    bus.write_byte_data(Device_Address, PWR_MGMT_1, 1)
    bus.write_byte_data(Device_Address, CONFIG, 0)
    bus.write_byte_data(Device_Address, 0x1B, 0)
    bus.write_byte_data(Device_Address, 0x1C, 0)
    bus.write_byte_data(Device_Address, INT_ENABLE, 1)

def low_pass_filter(prev_x, x, alpha=0.9):
    return alpha * prev_x + (1 - alpha) * x

def high_pass_filter(prev_y, y, alpha=0.9):
    return alpha * (prev_y + y)

def read_raw_data(addr):
    high = bus.read_byte_data(Device_Address, addr)
    low = bus.read_byte_data(Device_Address, addr + 1)
    value = ((high << 8) | low)
    if value > 32768:
        value = value - 65536
    return value

bus = smbus.SMBus(1)
Device_Address = 0x68
MPU_Init()
print("Reading Data of Gyroscope and Accelerometer")

imu_data = []
velocity = np.zeros(3)
position = np.zeros(3)
vel_prev = np.zeros(3)
Gx_prev, Gy_prev, Gz_prev = 0, 0, 0
Ax_prev, Ay_prev, Az_prev = 0, 0, 0

# Create an array to hold time values
time_data = []

for idx in range(50):
    acc_x = read_raw_data(ACCEL_XOUT_H)
    acc_y = read_raw_data(ACCEL_YOUT_H)
    acc_z = read_raw_data(ACCEL_ZOUT_H)
    gyro_x = read_raw_data(GYRO_XOUT_H)
    gyro_y = read_raw_data(GYRO_YOUT_H)
    gyro_z = read_raw_data(GYRO_ZOUT_H)

    Ax = acc_x / 16384.0 * 9.81
    Ay = acc_y / 16384.0 * 9.81
    Az = acc_z / 16384.0 * 9.81
    Gx = gyro_x / 131.0
    Gy = gyro_y / 131.0
    Gz = gyro_z / 131.0
    
    if idx > 0:
        Gx = low_pass_filter(Gx_prev, Gx)
        Gy = low_pass_filter(Gy_prev, Gy)
        Gz = low_pass_filter(Gz_prev, Gz)
        Ax = low_pass_filter(Ax_prev, Ax)
        Ay = low_pass_filter(Ay_prev, Ay)
        Az = low_pass_filter(Az_prev, Az)
    
    Gx_prev, Gy_prev, Gz_prev = Gx, Gy, Gz
    Ax_prev, Ay_prev, Az_prev = Ax, Ay, Az
    
    acceleration = np.array([Ax, Ay, Az])
    velocity = high_pass_filter(vel_prev, velocity + acceleration * DT, ALPHA)
    position += velocity * DT
    vel_prev = velocity.copy()
    
    # Append the time and sensor data
    imu_data.append([Gx, Gy, Gz, Ax, Ay, Az, velocity[0], velocity[1], velocity[2], position[0], position[1], position[2]])
    time_data.append(idx * DT)  # Time in seconds
    sleep(DT)

imu_data_arr = np.array(imu_data)
np.save('/home/toon/data/imu_data/imu_data.npy', imu_data_arr)

# Convert time_data to a numpy array
time_data_arr = np.array(time_data)

# Plot the data with time on the x-axis
fig, ax = plt.subplots(5, 1, figsize=(10, 12))

# Gyroscope data
ax[0].plot(time_data_arr, imu_data_arr[:, 0], label='Gx')
ax[0].plot(time_data_arr, imu_data_arr[:, 1], label='Gy')
ax[0].plot(time_data_arr, imu_data_arr[:, 2], label='Gz')
ax[0].set_title('Gyroscope data')
ax[0].set_xlabel('Time (s)')
ax[0].set_ylabel('Angular velocity (deg/s)')
ax[0].legend()

# Accelerometer data
ax[1].plot(time_data_arr, imu_data_arr[:, 3], label='Ax')
ax[1].plot(time_data_arr, imu_data_arr[:, 4], label='Ay')
ax[1].plot(time_data_arr, imu_data_arr[:, 5], label='Az')
ax[1].set_title('Accelerometer data')
ax[1].set_xlabel('Time (s)')
ax[1].set_ylabel('Acceleration (m/s^2)')
ax[1].legend()

# Velocity data
ax[2].plot(time_data_arr, imu_data_arr[:, 6], label='Vx')
ax[2].plot(time_data_arr, imu_data_arr[:, 7], label='Vy')
ax[2].set_title('Velocity data')
ax[2].set_xlabel('Time (s)')
ax[2].set_ylabel('Velocity (m/s)')
ax[2].legend()

# X Position plot
ax[3].plot(time_data_arr, imu_data_arr[:, 9], label='X position')
ax[3].set_title('X Position data')
ax[3].set_xlabel('Time (s)')
ax[3].set_ylabel('X Position (m)')
ax[3].legend()

# Y Position plot
ax[4].plot(time_data_arr, imu_data_arr[:, 10], label='Y position')
ax[4].set_title('Y Position data')
ax[4].set_xlabel('Time (s)')
ax[4].set_ylabel('Y Position (m)')
ax[4].legend()

plt.tight_layout()
plt.savefig('/home/toon/data/imu_data/imu_data.png')
plt.show()