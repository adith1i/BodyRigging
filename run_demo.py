#!/usr/bin/env python3
import cv2
import time
import numpy as np
import argparse
import os
from pose_detector import PoseDetector
from skeleton_visualizer import SkeletonVisualizer
from motion_analyzer import MotionAnalyzer
from smoothing import LandmarkSmoother
from utils import calculate_fps, draw_fps

def parse_args():
    parser = argparse.ArgumentParser(description='Body Rigging Demo Application')
    parser.add_argument('--video', type=str, required=True,
                        help='Path to video file')
    parser.add_argument('--output', type=str, default='',
                        help='Output video file (optional)')
    parser.add_argument('--width', type=int, default=1280,
                        help='Output width (default: 1280)')
    parser.add_argument('--height', type=int, default=720,
                        help='Output height (default: 720)')
    parser.add_argument('--detection_confidence', type=float, default=0.5,
                        help='Pose detection confidence threshold (default: 0.5)')
    parser.add_argument('--tracking_confidence', type=float, default=0.5,
                        help='Pose tracking confidence threshold (default: 0.5)')
    parser.add_argument('--visualization', type=str, default='skeleton',
                        choices=['skeleton', 'mesh', 'contour'],
                        help='Visualization type (default: skeleton)')
    parser.add_argument('--smoothing', type=str, default='EMA',
                        choices=['EMA', 'ONE_EURO', 'KALMAN', 'MOVING_AVG', 'NONE'],
                        help='Smoothing method (default: EMA)')
    parser.add_argument('--analyzer', action='store_true',
                        help='Enable motion analysis overlay')
    parser.add_argument('--speed', type=float, default=1.0,
                        help='Playback speed multiplier (default: 1.0)')
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()

    # Check if video file exists
    if not os.path.isfile(args.video):
        print(f"Error: Video file '{args.video}' not found.")
        return

    # Initialize video capture
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Video loaded: {args.video}")
    print(f"Resolution: {width}x{height}, FPS: {fps}, Frames: {frame_count}")

    # Set the visualization mode
    visualization_mode = {'skeleton': 0, 'mesh': 1, 'contour': 2}[args.visualization]

    # Initialize video writer if output specified
    video_writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_writer = cv2.VideoWriter(
            args.output, fourcc, fps, (args.width, args.height))
        print(f"Recording output to: {args.output}")

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

    # Initialize FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    processing_fps = 0

    # Calculate frame delay for specified speed
    frame_delay = max(1, int(1000 / (fps * args.speed)))

    # Progress tracking
    start_time = time.time()
    frames_processed = 0

    print("Starting processing...")
    print("Press 'ESC' to exit, 'SPACE' to pause/resume")

    # Main processing loop
    paused = False
    while True:
        if not paused:
            # Read frame from video
            ret, frame = cap.read()
            if not ret:
                print("End of video reached.")
                break

            # Resize frame if needed
            if frame.shape[1] != args.width or frame.shape[0] != args.height:
                frame = cv2.resize(frame, (args.width, args.height))

            # Process frame for pose detection
            pose_results = pose_detector.detect_pose(frame)

            # Apply additional landmark smoothing if enabled
            if landmark_smoother is not None and pose_results.pose_landmarks:
                pose_results.pose_landmarks = landmark_smoother.smooth_landmarks(
                    pose_results.pose_landmarks, time.time())

            # Update motion analyzer if enabled
            if motion_analyzer is not None and pose_results.pose_landmarks:
                motion_analyzer.update(pose_results.pose_landmarks, args.width, args.height)

            # Visualize pose if detected
            if pose_results.pose_landmarks:
                # Draw skeleton based on specified visualization mode
                if visualization_mode == 0:
                    frame = skeleton_visualizer.draw_skeleton(frame, pose_results)
                elif visualization_mode == 1:
                    frame = skeleton_visualizer.draw_mesh(frame, pose_results)
                elif visualization_mode == 2:
                    frame = skeleton_visualizer.draw_contour(frame, pose_results)

                # Display joint angles if analyzer is enabled
                if args.analyzer and motion_analyzer is not None:
                    frame = motion_analyzer.draw_angles(frame,
                                                     ['elbow_left', 'elbow_right',
                                                      'knee_left', 'knee_right'])

            # Calculate and display processing FPS
            processing_fps, fps_start_time, fps_frame_count = calculate_fps(
                fps_start_time, fps_frame_count)
            frame = draw_fps(frame, processing_fps)

            # Display progress
            frames_processed += 1
            progress = (frames_processed / frame_count) * 100
            elapsed_time = time.time() - start_time
            estimated_total = elapsed_time / (progress / 100) if progress > 0 else 0
            remaining_time = estimated_total - elapsed_time

            progress_text = f"Progress: {progress:.1f}% | Time remaining: {remaining_time:.1f}s"
            cv2.putText(frame, progress_text, (10, args.height - 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
            cv2.putText(frame, progress_text, (10, args.height - 20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            # Write frame to output video if enabled
            if video_writer is not None:
                video_writer.write(frame)

            # Display the frame
            cv2.imshow("Body Rigging Demo", frame)

        # Handle keyboard input with specified delay
        key = cv2.waitKey(frame_delay if not paused else 0) & 0xFF

        # ESC - exit
        if key == 27:
            break
        # Space - pause/resume
        elif key == 32:
            paused = not paused
            print("Playback", "paused" if paused else "resumed")

    # Release resources
    cap.release()
    if video_writer is not None:
        video_writer.release()
    cv2.destroyAllWindows()

    # Print summary
    total_time = time.time() - start_time
    print(f"Processing completed.")
    print(f"Total frames processed: {frames_processed}")
    print(f"Total processing time: {total_time:.2f} seconds")
    print(f"Average processing FPS: {frames_processed / total_time:.2f}")

    if args.output:
        print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main()
