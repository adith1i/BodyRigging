import cv2
import mediapipe as mp
import numpy as np

class SkeletonVisualizer:
    """Handles visualization of the skeleton and related elements."""

    def __init__(self, width, height):
        """Initialize the visualizer.

        Args:
            width: Width of the video frame
            height: Height of the video frame
        """
        self.width = width
        self.height = height
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_pose = mp.solutions.pose
        self.mp_drawing_styles = mp.solutions.drawing_styles

        # Define skeleton connections for different body parts
        self.POSE_CONNECTIONS = self.mp_pose.POSE_CONNECTIONS

        # Define body part color map (RGB)
        self.body_color_map = {
            'head': (255, 0, 0),      # Red
            'torso': (0, 255, 0),     # Green
            'right_arm': (0, 0, 255), # Blue
            'left_arm': (255, 255, 0),# Yellow
            'right_leg': (0, 255, 255),# Cyan
            'left_leg': (255, 0, 255) # Magenta
        }

        # Define joint color (BGR)
        self.joint_color = (255, 255, 255)  # White

        # Define joint radius
        self.joint_radius = 5

        # Define line thickness
        self.line_thickness = 2

        # Setup connections for each body part
        self.head_connections = [
            (0, 1), (1, 2), (2, 3), (3, 7),
            (0, 4), (4, 5), (5, 6), (6, 8),
            (9, 10)
        ]

        self.torso_connections = [
            (11, 12), (11, 23), (12, 24), (23, 24)
        ]

        self.left_arm_connections = [
            (11, 13), (13, 15), (15, 17), (15, 19), (15, 21)
        ]

        self.right_arm_connections = [
            (12, 14), (14, 16), (16, 18), (16, 20), (16, 22)
        ]

        self.left_leg_connections = [
            (23, 25), (25, 27), (27, 29), (27, 31)
        ]

        self.right_leg_connections = [
            (24, 26), (26, 28), (28, 30), (28, 32)
        ]

        # Motion trails - store previous positions of key joints
        self.trail_length = 20
        self.trails = {
            'left_wrist': [],
            'right_wrist': [],
            'left_ankle': [],
            'right_ankle': []
        }

    def draw_skeleton(self, image, results):
        """Draw the skeleton on the image.

        Args:
            image: The image frame to draw on
            results: The pose landmarks from MediaPipe

        Returns:
            The image with the skeleton drawn on it
        """
        if not results.pose_landmarks:
            return image

        # Make a copy of the image to avoid modifying the original
        output_image = image.copy()

        # Get landmarks
        landmarks = results.pose_landmarks.landmark

        # Update motion trails
        self._update_motion_trails(landmarks)

        # Draw motion trails
        self._draw_motion_trails(output_image)

        # Draw connections for different body parts with specific colors
        self._draw_connections(output_image, landmarks, self.head_connections,
                              self.body_color_map['head'])
        self._draw_connections(output_image, landmarks, self.torso_connections,
                              self.body_color_map['torso'])
        self._draw_connections(output_image, landmarks, self.left_arm_connections,
                              self.body_color_map['left_arm'])
        self._draw_connections(output_image, landmarks, self.right_arm_connections,
                              self.body_color_map['right_arm'])
        self._draw_connections(output_image, landmarks, self.left_leg_connections,
                              self.body_color_map['left_leg'])
        self._draw_connections(output_image, landmarks, self.right_leg_connections,
                              self.body_color_map['right_leg'])

        # Draw joints (landmarks)
        for idx, landmark in enumerate(landmarks):
            # Convert landmark position to pixel coordinates
            cx, cy = int(landmark.x * self.width), int(landmark.y * self.height)

            # Draw joint
            cv2.circle(output_image, (cx, cy), self.joint_radius, self.joint_color, -1)

        return output_image

    def draw_mesh(self, image, results):
        """Draw a mesh representation of the pose.

        Args:
            image: The image frame to draw on
            results: The pose landmarks from MediaPipe

        Returns:
            The image with mesh visualization
        """
        if not results.pose_landmarks:
            return image

        # Make a copy of the image
        output_image = image.copy()

        # Use MediaPipe's built-in drawing utilities with custom styles
        custom_drawing_spec = self.mp_drawing.DrawingSpec(
            color=(0, 255, 0),  # Green
            thickness=1,
            circle_radius=1
        )

        self.mp_drawing.draw_landmarks(
            output_image,
            results.pose_landmarks,
            self.POSE_CONNECTIONS,
            landmark_drawing_spec=custom_drawing_spec,
            connection_drawing_spec=self.mp_drawing_styles.get_default_pose_connections_style()
        )

        # If segmentation mask is available, use it
        if hasattr(results, 'segmentation_mask') and results.segmentation_mask is not None:
            # Apply a semi-transparent overlay using the segmentation mask
            segmentation_mask = results.segmentation_mask

            # Resize if needed
            if segmentation_mask.shape[0] != self.height or segmentation_mask.shape[1] != self.width:
                segmentation_mask = cv2.resize(
                    segmentation_mask, (self.width, self.height))

            # Create a colored overlay for the mask
            condition = np.stack((segmentation_mask,) * 3, axis=-1) > 0.1

            # Fill the mask with a green tint
            bg_image = np.zeros(output_image.shape, dtype=np.uint8)
            bg_image[:] = (0, 255, 0)  # Green background

            # Blend based on the mask condition
            output_image = np.where(condition,
                                  cv2.addWeighted(output_image, 0.7, bg_image, 0.3, 0),
                                  output_image)

        return output_image

    def draw_contour(self, image, results):
        """Draw a contour outline of the pose.

        Args:
            image: The image frame to draw on
            results: The pose landmarks from MediaPipe

        Returns:
            The image with contour visualization
        """
        if not results.pose_landmarks:
            return image

        # Make a copy of the image
        output_image = image.copy()

        # Get landmarks
        landmarks = results.pose_landmarks.landmark

        # Create arrays for different body parts
        head_points = []
        left_arm_points = []
        right_arm_points = []
        torso_points = []
        left_leg_points = []
        right_leg_points = []

        # Head contour (simplified)
        for idx in [0, 7, 8, 10, 9]:
            x, y = int(landmarks[idx].x * self.width), int(landmarks[idx].y * self.height)
            head_points.append([x, y])

        # Left arm
        for idx in [11, 13, 15, 17, 19, 15, 21]:
            x, y = int(landmarks[idx].x * self.width), int(landmarks[idx].y * self.height)
            left_arm_points.append([x, y])

        # Right arm
        for idx in [12, 14, 16, 18, 20, 16, 22]:
            x, y = int(landmarks[idx].x * self.width), int(landmarks[idx].y * self.height)
            right_arm_points.append([x, y])

        # Torso
        for idx in [11, 12, 24, 23]:
            x, y = int(landmarks[idx].x * self.width), int(landmarks[idx].y * self.height)
            torso_points.append([x, y])

        # Left leg
        for idx in [23, 25, 27, 29, 31]:
            x, y = int(landmarks[idx].x * self.width), int(landmarks[idx].y * self.height)
            left_leg_points.append([x, y])

        # Right leg
        for idx in [24, 26, 28, 30, 32]:
            x, y = int(landmarks[idx].x * self.width), int(landmarks[idx].y * self.height)
            right_leg_points.append([x, y])

        # Draw filled contours
        if len(head_points) > 2:
            cv2.fillPoly(output_image, [np.array(head_points)], self.body_color_map['head'])
        if len(torso_points) > 2:
            cv2.fillPoly(output_image, [np.array(torso_points)], self.body_color_map['torso'])
        if len(left_arm_points) > 2:
            cv2.fillPoly(output_image, [np.array(left_arm_points)], self.body_color_map['left_arm'])
        if len(right_arm_points) > 2:
            cv2.fillPoly(output_image, [np.array(right_arm_points)], self.body_color_map['right_arm'])
        if len(left_leg_points) > 2:
            cv2.fillPoly(output_image, [np.array(left_leg_points)], self.body_color_map['left_leg'])
        if len(right_leg_points) > 2:
            cv2.fillPoly(output_image, [np.array(right_leg_points)], self.body_color_map['right_leg'])

        # Draw outline around the entire body
        all_points = (head_points + torso_points + left_arm_points +
                     right_arm_points + left_leg_points + right_leg_points)

        if len(all_points) > 0:
            # Use convex hull to get the outer contour
            hull = cv2.convexHull(np.array(all_points))
            cv2.polylines(output_image, [hull], True, (255, 255, 255), 2)

        return output_image

    def label_joints(self, image, results):
        """Label the joints with their names.

        Args:
            image: The image frame to draw on
            results: The pose landmarks from MediaPipe

        Returns:
            The image with labeled joints
        """
        if not results.pose_landmarks:
            return image

        # Make a copy of the image
        output_image = image.copy()

        # Get landmarks
        landmarks = results.pose_landmarks.landmark

        # Define joint names mapping to indices
        joint_names = {
            0: "NOSE",
            11: "L_SHLDR",
            12: "R_SHLDR",
            13: "L_ELBOW",
            14: "R_ELBOW",
            15: "L_WRIST",
            16: "R_WRIST",
            23: "L_HIP",
            24: "R_HIP",
            25: "L_KNEE",
            26: "R_KNEE",
            27: "L_ANKLE",
            28: "R_ANKLE"
        }

        # Label selected joints
        for idx, name in joint_names.items():
            if idx < len(landmarks):
                x, y = int(landmarks[idx].x * self.width), int(landmarks[idx].y * self.height)
                cv2.putText(output_image, name, (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(output_image, name, (x, y - 10),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return output_image

    def _draw_connections(self, image, landmarks, connections, color):
        """Draw connections between landmarks.

        Args:
            image: The image to draw on
            landmarks: The pose landmarks
            connections: List of tuples defining connections
            color: RGB color for the connections
        """
        for connection in connections:
            start_idx, end_idx = connection

            if start_idx < len(landmarks) and end_idx < len(landmarks):
                start_point = (int(landmarks[start_idx].x * self.width),
                              int(landmarks[start_idx].y * self.height))
                end_point = (int(landmarks[end_idx].x * self.width),
                            int(landmarks[end_idx].y * self.height))

                cv2.line(image, start_point, end_point, color, self.line_thickness)

    def _update_motion_trails(self, landmarks):
        """Update the motion trails for key joints.

        Args:
            landmarks: The pose landmarks
        """
        # Map joint names to their landmark indices
        trail_indices = {
            'left_wrist': 15,
            'right_wrist': 16,
            'left_ankle': 27,
            'right_ankle': 28
        }

        # Update each trail
        for joint_name, idx in trail_indices.items():
            if idx < len(landmarks):
                x, y = int(landmarks[idx].x * self.width), int(landmarks[idx].y * self.height)
                self.trails[joint_name].append((x, y))

                # Limit trail length
                if len(self.trails[joint_name]) > self.trail_length:
                    self.trails[joint_name].pop(0)

    def _draw_motion_trails(self, image):
        """Draw motion trails for tracked joints.

        Args:
            image: The image to draw on
        """
        # Trail colors (BGR)
        trail_colors = {
            'left_wrist': (255, 255, 0),
            'right_wrist': (0, 255, 255),
            'left_ankle': (255, 0, 255),
            'right_ankle': (255, 128, 0)
        }

        # Draw each trail
        for joint_name, trail in self.trails.items():
            if len(trail) < 2:
                continue

            # Draw lines with decreasing opacity
            for i in range(1, len(trail)):
                # Calculate opacity based on position in trail
                opacity = int(255 * (i / len(trail)))

                # Get the base color
                color = trail_colors[joint_name]

                # Draw the line segment
                cv2.line(image, trail[i-1], trail[i], color, 2)

    def calculate_angle(self, p1, p2, p3):
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

        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        angle = np.arccos(np.clip(cosine_angle, -1.0, 1.0))

        return np.degrees(angle)
