import os
display_window = os.environ.get('DISPLAY')
os.environ['DISPLAY'] = ''
import cv2
import time
import json
import numpy as np
import multiprocessing as mp
mp.set_start_method('spawn', force=True)

import spatialmath as sm
import pyrealsense2 as rs
from klampt.control import TimedLooper
from scipy.spatial.transform import Rotation as R

from ai4ia_control.utils.conversions import trv2SE3, SE32trv
from ai4ia_hardware_interfaces.robot_system import RobotSystem
from ai4ia_control.trajectory.trajectory_primitives import target_twist_SE3_trajectory

from pose_estimation_stream import PoseEstimationStream
from pose_estimation_stream_utils import IoU, Display, Pose_Estimation_Writer, Pose_Estimation_Ploter

def get_camera_conf():
    # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    # Enable the streams
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    return pipeline, config

class robot_track:

    def _init_(self, robot, max_v= 20/1000, max_w = np.deg2rad(10), steps=10, dt=0.2, alpha=0.1, min_t_norm = 1/1000, min_R_norm= np.deg2rad(5)):

        self.robot = robot
        self.max_v = max_v
        self.max_w = max_w
        self.steps = steps
        self.dt = dt
        self.alpha = alpha
        self.min_t_norm = min_t_norm
        self.min_R_norm = min_R_norm
        self.V_prev = np.zeros(6)
        self.is_move = False
        self.T_base_plug_list = []
    
    def track(self, trv_pose, trv_goal):

        if trv_pose is None:
            self.stop()
        else:
            t_pose = trv_pose[:3]
            R_pose = R.from_quat(trv_pose[3:]).as_rotvec()
            trv_pose = np.append(t_pose, R_pose)
            
            T_pose = trv2SE3(trv_pose)
            T_goal = trv2SE3(trv_goal)

            T_pose2goal =  T_pose @ T_goal.inv()

            t_pose2goal = T_pose2goal.t
            R_pose2goal = R.from_matrix(T_pose2goal.R).as_euler('zxy')

            for ii in range(len(R_pose2goal)):
                if R_pose2goal[ii] > np.pi/2:
                    R_pose2goal[ii] = R_pose2goal[ii] - np.pi
                    print('angle {} was {} and normlized to {}'.format(ii, np.rad2deg(R_pose2goal[ii]+np.pi), np.rad2deg(R_pose2goal[ii])))

                elif R_pose2goal[ii] < -np.pi/2:
                    R_pose2goal[ii] = R_pose2goal[ii] + np.pi
                    print('angle {} was {} and normlized to {}'.format(ii, np.rad2deg(R_pose2goal[ii]-np.pi), np.rad2deg(R_pose2goal[ii])))

            t_pose2goal = np.round(t_pose2goal, decimals=4)
            R_pose2goal = np.deg2rad(np.round(np.rad2deg(R_pose2goal), decimals=1))

            print(t_pose2goal*1000, '[mm]', np.rad2deg(R_pose2goal), '[deg]')

            trv_pose2goal = np.append(t_pose2goal, R.from_euler('zxy', R_pose2goal).as_rotvec())
            trv_state = self.robot.manipulator_interface.state.X
            T_state = trv2SE3(trv_state)

            trv_pose2goal = np.append(T_state.R @ trv_pose2goal[:3], T_state.R @ trv_pose2goal[3:])

            t_norm = np.linalg.norm(trv_pose2goal[:3])
            R_norm = np.linalg.norm(trv_pose2goal[3:])

            if t_norm < self.min_t_norm and R_norm < self.min_R_norm:
                self.stop()

            else:
                if t_norm < 0.1/1000:
                    t_V_d = np.zeros(3)
                else:
                    t_V_d = trv_pose2goal[:3]/t_norm
                if R_norm < 0.1:
                    R_V_d = np.zeros(3)
                else:
                    R_V_d = trv_pose2goal[3:]/R_norm

                trv_V_d = np.append(t_V_d, R_V_d)

                v_norm = np.min([self.max_v, self.alpha*(t_norm/self.dt)])
                w_norm = np.min([self.max_w, self.alpha*(R_norm/self.dt)])

                V = np.append(trv_V_d[:3]*v_norm, trv_V_d[3:]*w_norm)

                traj = target_twist_SE3_trajectory(trv_state, self.V_prev, V, np.linspace(0, self.dt, self.steps))
                self.robot.manipulator_interface.track_SE3_trajectory(traj)
                self.V_prev = V
                self.is_move = True

    def stop(self):
        if self.is_move:
            self.robot.manipulator_interface.deactivate_controller()
            print("Deactivating controller")
            self.V_prev = np.zeros(6)
            self.is_move = False

def robot_track_pose(shared_dict):

    root = os.path.dirname(os.path.abspath(_file_))

    config_file = os.path.join('/', *root.split(os.sep)[:-4], 'ai4ia_hardware_interfaces', 'configs', 'ur5e_uri.yaml')

    robot = RobotSystem.from_config(config_file)
    robot.run()
    robot.manipulator_interface.zero_ft_sensor()
    robot_tracker = robot_track(robot)

    trv_goal = [0, 6/1000, 0, np.pi, 0, 0]
    # trv_goal = [-0.00223456, -0.00836555, -0.00463823, -0.05208202,  0.14087745,  0.30300121]

    dt = 0.5
    policy_loop = TimedLooper(dt=dt, name="Policy")

    try:
        while shared_dict["run"]:
            if shared_dict["follow_flag"]:
                if policy_loop:
                        robot_tracker.track(trv_pose = shared_dict["trv_pose"], trv_goal = trv_goal)
            else:
                robot_tracker.stop()

    finally:
        robot.stop()

def pose_estimation_stream(segment_anything_2_dir, megapose6d_dir, initial_frame = None, points = [], labels = []):

    # Start streaming
    pipeline, config = get_camera_conf()
    _ = pipeline.start(config)

    # Retrieve camera intrinsics from the active profile
    profile = pipeline.get_active_profile()
    color_sensor = profile.get_device().query_sensors()[1]
    color_sensor.set_option(rs.option.exposure, 800)
    color_sensor.set_option(rs.option.gain, 60)

    root = os.path.dirname(os.path.abspath(_file_))

    pose_estimator = PoseEstimationStream(segment_anything_2_dir, megapose6d_dir)

    display = Display(display_window)

    ploter = Pose_Estimation_Ploter(
        CAD_path = os.path.join(root, 'CAD_models', 'connector.obj'),
        intrinsics_path = os.path.join(root, 'camera_intrinsics.json'),
        t_cam2gripper_path = os.path.join(root, 'T_cam2gripper.json'),
        number_of_points = 5000
        )
    
    if display.display_window is not None:
        # Callback for adding propts via mouse
        def mouse_callback(event, x, y, flags, param):
            # Check if the left mouse button is clicked
            if event == cv2.EVENT_LBUTTONDOWN:
                points.append([x,y])
                labels.append([1])
                pose_estimator.SAM2.set_initial_frame(initial_frame, points, labels)
                masked_frame = pose_estimator.SAM2.get_masked_frame(disply_mode=2)
                cv2.circle(masked_frame, (x,y), 3, (0, 255, 0), -1)             
                cv2.imshow('SAM2 RGB', masked_frame)
                
            # Check if the right mouse button is clicked
            elif event == cv2.EVENT_RBUTTONDOWN:
                points.append([x,y])
                labels.append([0])
                pose_estimator.SAM2.set_initial_frame(initial_frame, points, labels)
                masked_frame = pose_estimator.SAM2.get_masked_frame(disply_mode=2)
                cv2.circle(masked_frame, (x,y), 3, (0, 0, 255), -1)
                cv2.imshow('SAM2 RGB', masked_frame)

        while len(points) == 0:
            
            # Load frame
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            frame = np.asanyarray(color_frame.get_data())
            initial_frame = frame.copy()

            with display:
                cv2.imshow('SAM2 RGB', initial_frame)
                cv2.setMouseCallback('SAM2 RGB', mouse_callback)
                key = cv2.waitKey(1)
                if key == ord('q') or key == ord('Q'):
                    break

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # models inititalization
    pose_estimator.models_initialization(frame = initial_frame, points=points, labels=labels)
    reset_pose = True

    with mp.Manager() as manager:
        shared_dict = manager.dict()
        shared_dict["trv_pose"] = None
        shared_dict["follow_flag"] = False
        shared_dict["run"] = True

        robot_tracking_process = mp.Process(target=robot_track_pose, args=(shared_dict,))
        robot_tracking_process.start()

        try:
            while shared_dict["run"]:
                
                # Load frame
                frame = pipeline.wait_for_frames()
                color_frame = frame.get_color_frame()
                if not color_frame:
                    continue
                frame = np.asanyarray(color_frame.get_data())

                start_time = time.time()

                pose, _ = pose_estimator.Pose_Estimation(frame, reset_pose=reset_pose)
                reset_pose=False

                end_time = time.time()
                # print("iteration time:",round((end_time-start_time),2),'sec')

                pose_im = ploter.show(frame, pose, calibrate_pose = True)


                with display:
                    if display.display_window is not None:
                        cv2.imshow('6D pose estimation',pose_im)
                        key = cv2.waitKey(1)
                        if key == ord('q') or key == ord('Q'):
                            shared_dict["run"] = False

                        if key == ord('f') or key == ord('F'):
                            if shared_dict["follow_flag"]:
                                print('Stop following the plug')
                                shared_dict["follow_flag"] = False
                            else:
                                print('Starting following the plug')
                                shared_dict["follow_flag"] = True

                iou = IoU(mask1 = ploter.mask , mask2 = pose_estimator.mask)

                if iou < 0.5:
                    reset_pose = True
                    trv_pose = None
                    print("restarting with coarse estimate")

                else:
                    trv_pose = ploter.calibrated_pose  

                shared_dict["trv_pose"] = trv_pose

        finally:
            
            pose_estimator.reset()   
            cv2.destroyAllWindows()
            pipeline.stop()
            robot_tracking_process.terminate()
            robot_tracking_process.join()

if _name_ == "_main_":

    segment_anything_2_dir = "./segment-anything-2"
    megapose6d_dir = "./megapose6d"

    pose_estimation_stream(segment_anything_2_dir, megapose6d_dir)