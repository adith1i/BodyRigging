import cv2
import numpy as np
import time
from collections import deque


class MotionAnalyzer:
    """Analyzes body movement patterns and metrics."""

    def __init__(self, history_length=60):
        """Initialize the motion analyzer.

        Args:
            history_length: Number of frames to keep in history
        """
        self.history_length = history_length

        # Initialize landmark history for tracking movement
        self.landmarks_history = deque(maxlen=history_length)

        # Joint angle history
        self.joint_angles = {
            'elbow_left': deque(maxlen=history_length),
            'elbow_right': deque(maxlen=history_length),
            'shoulder_left': deque(maxlen=history_length),
            'shoulder_right': deque(maxlen=history_length),
            'hip_left': deque(maxlen=history_length),
            'hip_right': deque(maxlen=history_length),
            'knee_left': deque(maxlen=history_length),
            'knee_right': deque(maxlen=history_length),
            'ankle_left': deque(maxlen=history_length),
            'ankle_right': deque(maxlen=history_length)
        }

        # Movement speed measurements
        self.speed_measurements = {
            'wrist_left': deque(maxlen=history_length),
            'wrist_right': deque(maxlen=history_length),
            'ankle_left': deque(maxlen=history_length),
            'ankle_right': deque(maxlen=history_length)
        }

        # Time of last frame for speed calculation
        self.last_frame_time = time.time()

    def update(self, pose_landmarks, img_width, img_height):
        """Update motion analysis with new pose data.

        Args:
            pose_landmarks: MediaPipe pose landmarks
            img_width: Image width
            img_height: Image height
        """
        if not pose_landmarks:
            return

        # Convert landmarks to pixel coordinates
        landmarks = []
        for landmark in pose_landmarks.landmark:
            x = int(landmark.x * img_width)
            y = int(landmark.y * img_height)
            landmarks.append((x, y))

        # Add landmarks to history
        self.landmarks_history.append(landmarks)

        # Calculate joint angles
        self._calculate_joint_angles(landmarks)

        # Calculate movement speeds
        current_time = time.time()
        time_diff = current_time - self.last_frame_time
        self._calculate_speeds(landmarks, time_diff)
        self.last_frame_time = current_time

    def _calculate_joint_angles(self, landmarks):
        """Calculate angles at key joints.

        Args:
            landmarks: List of landmark coordinates
        """
        # Map of joints to the three points that define the angle
        # Format: joint_name: (point1_idx, vertex_idx, point2_idx)
        joint_definitions = {
            'elbow_left': (11, 13, 15),   # shoulder - elbow - wrist
            'elbow_right': (12, 14, 16),  # shoulder - elbow - wrist
            'shoulder_left': (13, 11, 23), # elbow - shoulder - hip
            'shoulder_right': (14, 12, 24), # elbow - shoulder - hip
            'hip_left': (11, 23, 25),    # shoulder - hip - knee
            'hip_right': (12, 24, 26),   # shoulder - hip - knee
            'knee_left': (23, 25, 27),   # hip - knee - ankle
            'knee_right': (24, 26, 28),  # hip - knee - ankle
            'ankle_left': (25, 27, 31),  # knee - ankle - foot
            'ankle_right': (26, 28, 32)  # knee - ankle - foot
        }

        # Calculate each joint angle
        for joint_name, (p1_idx, p2_idx, p3_idx) in joint_definitions.items():
            # Skip if any of the landmarks are missing
            if (p1_idx >= len(landmarks) or
                p2_idx >= len(landmarks) or
                p3_idx >= len(landmarks)):
                continue

            p1 = landmarks[p1_idx]
            p2 = landmarks[p2_idx]
            p3 = landmarks[p3_idx]

            angle = self._calculate_angle(p1, p2, p3)
            self.joint_angles[joint_name].append(angle)

    def _calculate_angle(self, p1, p2, p3):
        """Calculate the angle between three points.

        Args:
            p1, p2, p3: Three points where p2 is the vertex

        Returns:
            Angle in degrees
        """
        a = np.array([p1[0], p1[1]])
        b = np.array([p2[0], p2[1]])
        c = np.array([p3[0], p3[1]])

        ba = a - b
        bc = c - b

        # Calculate dot product and normalize
        dot_product = np.dot(ba, bc)
        norm_ba = np.linalg.norm(ba)
        norm_bc = np.linalg.norm(bc)

        # Prevent division by zero
        if norm_ba == 0 or norm_bc == 0:
            return 0

        cosine_angle = dot_product / (norm_ba * norm_bc)

        # Ensure value is in valid range for arccos
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)

        angle = np.arccos(cosine_angle)

        return np.degrees(angle)

    def _calculate_speeds(self, landmarks, time_diff):
        """Calculate movement speeds for specific joints.

        Args:
            landmarks: List of landmark coordinates
            time_diff: Time difference between frames
        """
        # Skip if no history yet
        if len(self.landmarks_history) < 2:
            return

        # Map of joint names to their landmark indices
        joint_indices = {
            'wrist_left': 15,
            'wrist_right': 16,
            'ankle_left': 27,
            'ankle_right': 28
        }

        # Previous landmarks
        prev_landmarks = self.landmarks_history[-2]

        # Calculate speed for each tracked joint
        for joint_name, idx in joint_indices.items():
            # Skip if index is out of bounds
            if idx >= len(landmarks) or idx >= len(prev_landmarks):
                continue

            # Current and previous positions
            curr_pos = landmarks[idx]
            prev_pos = prev_landmarks[idx]

            # Calculate displacement (in pixels)
            dx = curr_pos[0] - prev_pos[0]
            dy = curr_pos[1] - prev_pos[1]
            distance = np.sqrt(dx*dx + dy*dy)

            # Calculate speed (pixels per second)
            if time_diff > 0:
                speed = distance / time_diff
                self.speed_measurements[joint_name].append(speed)
            else:
                self.speed_measurements[joint_name].append(0)

    def get_joint_angle(self, joint_name):
        """Get the current angle for a specific joint.

        Args:
            joint_name: Name of the joint

        Returns:
            Current angle in degrees, or None if not available
        """
        if joint_name in self.joint_angles and self.joint_angles[joint_name]:
            return self.joint_angles[joint_name][-1]
        return None

    def get_speed(self, joint_name):
        """Get the current speed for a specific joint.

        Args:
            joint_name: Name of the joint

        Returns:
            Current speed in pixels per second, or None if not available
        """
        if joint_name in self.speed_measurements and self.speed_measurements[joint_name]:
            return self.speed_measurements[joint_name][-1]
        return None

    def draw_angles(self, image, joint_list=None):
        """Draw joint angles on the image.

        Args:
            image: Image to draw on
            joint_list: List of joint names to display, or None for all

        Returns:
            Image with angles drawn
        """
        # Show all joints if none specified
        if joint_list is None:
            joint_list = list(self.joint_angles.keys())

        # Position for angle display
        x_pos = 10
        y_pos = 70  # Start below the FPS counter

        # Make a copy of the image
        output_image = image.copy()

        # Draw each joint angle
        for joint_name in joint_list:
            angle = self.get_joint_angle(joint_name)
            if angle is not None:
                text = f"{joint_name}: {angle:.1f}°"
                cv2.putText(output_image, text, (x_pos, y_pos),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(output_image, text, (x_pos, y_pos),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_pos += 20

        return output_image

    def draw_movement_heatmap(self, image, alpha=0.4):
        """Draw a movement heatmap based on historical data.

        Args:
            image: Image to draw on
            alpha: Transparency level

        Returns:
            Image with movement heatmap overlay
        """
        # Skip if not enough history
        if len(self.landmarks_history) < 2:
            return image

        # Create a blank heatmap
        heatmap = np.zeros_like(image)

        # Track points of interest (wrists and ankles)
        track_indices = [15, 16, 27, 28]  # Left wrist, right wrist, left ankle, right ankle

        # Iterate through history and add movement to heatmap
        for i in range(1, len(self.landmarks_history)):
            prev_landmarks = self.landmarks_history[i-1]
            curr_landmarks = self.landmarks_history[i]

            for idx in track_indices:
                if idx < len(prev_landmarks) and idx < len(curr_landmarks):
                    # Get previous and current positions
                    prev_pos = prev_landmarks[idx]
                    curr_pos = curr_landmarks[idx]

                    # Draw line on heatmap
                    cv2.line(heatmap, prev_pos, curr_pos, (0, 255, 0), 2)

        # Apply Gaussian blur to make it look more like a heatmap
        heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)

        # Overlay heatmap on original image
        overlay = cv2.addWeighted(image, 1.0, heatmap, alpha, 0)

        return overlay
