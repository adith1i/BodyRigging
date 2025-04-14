#!/usr/bin/env python3
import cv2
import time
import numpy as np
import argparse
from pose_detector import PoseDetector
from skeleton_visualizer import SkeletonVisualizer
from motion_analyzer import MotionAnalyzer
from smoothing import LandmarkSmoother
from utils import calculate_fps, draw_fps
from ui.control_panel import ControlPanel
from ui.data_display import DataDisplay

def parse_args():
    parser = argparse.ArgumentParser(description='Body Rigging and Tracking Application')
    parser.add_argument('--camera', type=int, default=0, help='Camera index (default: 0)')
    parser.add_argument('--width', type=int, default=1280, help='Camera width (default: 1280)')
    parser.add_argument('--height', type=int, default=720, help='Camera height (default: 720)')
    parser.add_argument('--flip', action='store_true', help='Flip camera horizontally')
    parser.add_argument('--detection_confidence', type=float, default=0.5,
                        help='Pose detection confidence threshold (default: 0.5)')
    parser.add_argument('--tracking_confidence', type=float, default=0.5,
                        help='Pose tracking confidence threshold (default: 0.5)')
    parser.add_argument('--ui', action='store_true', help='Enable UI controls and data display')
    parser.add_argument('--analyzer', action='store_true', help='Enable motion analysis')
    parser.add_argument('--smoothing', type=str, default='EMA',
                        choices=['EMA', 'ONE_EURO', 'KALMAN', 'MOVING_AVG', 'NONE'],
                        help='Smoothing method (default: EMA)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()

    # Initialize camera
    cap = cv2.VideoCapture(args.camera)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Check if camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    # Get actual camera dimensions
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera initialized with resolution: {width}x{height}")

    # Initialize pose detector
    pose_detector = PoseDetector(
        detection_confidence=args.detection_confidence,
        tracking_confidence=args.tracking_confidence
    )

    # Initialize skeleton visualizer
    skeleton_visualizer = SkeletonVisualizer(width, height)

    # Initialize landmark smoother if not NONE
    landmark_smoother = None
    if args.smoothing != 'NONE':
        landmark_smoother = LandmarkSmoother()
        landmark_smoother.set_smoothing_method(args.smoothing)

    # Initialize motion analyzer if enabled
    motion_analyzer = None
    if args.analyzer:
        motion_analyzer = MotionAnalyzer()

    # Initialize UI components if enabled
    control_panel = None
    data_display = None
    if args.ui:
        control_panel = ControlPanel()
        data_display = DataDisplay()

    # Initialize FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    fps = 0

    # Main application loop
    paused = False
    visualization_mode = 0  # 0: skeleton, 1: mesh, 2: contour
    smooth_enabled = True

    print("Body Rigging Application started.")
    print("Controls:")
    print("  ESC: Exit")
    print("  Space: Toggle visualization mode")
    print("  S: Toggle smoothing")
    print("  P: Pause/Resume")
    print("  R: Reset tracking")

    while True:
        # Skip frame processing if paused
        if not paused:
            # Read frame from camera
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image")
                break

            # Flip frame horizontally if requested
            if args.flip:
                frame = cv2.flip(frame, 1)

            # Process frame for pose detection
            pose_results = pose_detector.detect_pose(frame, smooth=smooth_enabled)

            # Apply additional landmark smoothing if enabled
            if landmark_smoother is not None and smooth_enabled and pose_results.pose_landmarks:
                pose_results.pose_landmarks = landmark_smoother.smooth_landmarks(
                    pose_results.pose_landmarks, time.time())

            # Update motion analyzer if enabled
            if motion_analyzer is not None and pose_results.pose_landmarks:
                motion_analyzer.update(pose_results.pose_landmarks, width, height)

            # Update data display if enabled
            if data_display is not None and motion_analyzer is not None:
                data_display.update(motion_analyzer)

            # If pose detected, visualize skeleton
            if pose_results.pose_landmarks:
                # Draw skeleton based on current visualization mode
                if visualization_mode == 0:
                    frame = skeleton_visualizer.draw_skeleton(frame, pose_results)
                elif visualization_mode == 1:
                    frame = skeleton_visualizer.draw_mesh(frame, pose_results)
                elif visualization_mode == 2:
                    frame = skeleton_visualizer.draw_contour(frame, pose_results)

                # Display joint angles if analyzer is enabled
                if motion_analyzer is not None:
                    frame = motion_analyzer.draw_angles(frame,
                                                        ['elbow_left', 'elbow_right',
                                                         'knee_left', 'knee_right'])

            # Calculate and display FPS
            fps, fps_start_time, fps_frame_count = calculate_fps(
                fps_start_time, fps_frame_count)
            frame = draw_fps(frame, fps)

            # Display application state
            mode_text = ["Skeleton", "Mesh", "Contour"][visualization_mode]
            smooth_text = "ON" if smooth_enabled else "OFF"
            cv2.putText(frame, f"Mode: {mode_text} | Smoothing: {smooth_text}",
                      (10, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                      0.6, (255, 255, 255), 2)

            # Display tracking frame
            cv2.imshow("Body Rigging", frame)

            # Show UI components if enabled
            if control_panel is not None:
                control_panel.show()

                # Get parameters from control panel
                params = control_panel.get_parameters()

                # Apply UI parameters
                visualization_mode = 0 if params['skeleton'] else 1 if params['mesh'] else 2
                smooth_enabled = params['smoothing'] > 0
                if landmark_smoother is not None:
                    landmark_smoother.ema_alpha = params['smoothing'] / 100.0

                # Handle recording button
                if params['record'] and data_display is not None:
                    if not data_display.recording:
                        data_display.start_recording()
                elif not params['record'] and data_display is not None and data_display.recording:
                    data_display.stop_recording()

                # Handle reset button
                if params['reset']:
                    pose_detector.reset()
                    if landmark_smoother is not None:
                        landmark_smoother.reset()
                    control_panel.buttons['reset']['active'] = False

            # Show data display if enabled
            if data_display is not None:
                data_view = data_display.create_data_view()
                cv2.imshow("Motion Analysis", data_view)

        # Process keyboard input
        key = cv2.waitKey(1) & 0xFF

        # ESC key - exit
        if key == 27:
            break
        # Space key - toggle visualization mode
        elif key == 32:
            visualization_mode = (visualization_mode + 1) % 3
        # 'S' key - toggle smoothing
        elif key == ord('s') or key == ord('S'):
            smooth_enabled = not smooth_enabled
        # 'P' key - toggle pause
        elif key == ord('p') or key == ord('P'):
            paused = not paused
        # 'R' key - reset tracking
        elif key == ord('r') or key == ord('R'):
            pose_detector.reset()
            if landmark_smoother is not None:
                landmark_smoother.reset()

    # Release resources
    cap.release()
    cv2.destroyAllWindows()

    # Save recording data if active
    if data_display is not None and data_display.recording:
        data_display.stop_recording()

    print("Application closed.")

if __name__ == "__main__":
    main()
