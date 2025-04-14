import cv2
import mediapipe as mp
import time
import numpy as np

class PoseDetector:
    """Handles pose detection and tracking using MediaPipe Pose solution."""

    def __init__(self,
                 static_image_mode=False,
                 model_complexity=1,
                 smooth_landmarks=True,
                 enable_segmentation=False,
                 smooth_segmentation=True,
                 detection_confidence=0.5,
                 tracking_confidence=0.5):
        """Initialize MediaPipe Pose components.

        Args:
            static_image_mode: Whether to process static images or video stream
            model_complexity: 0, 1, or 2. Higher is more accurate but slower
            smooth_landmarks: Whether to filter landmarks to reduce jitter
            enable_segmentation: Whether to generate segmentation mask
            smooth_segmentation: Whether to filter segmentation mask
            detection_confidence: Min confidence for person detection
            tracking_confidence: Min confidence for pose tracking
        """
        self.static_image_mode = static_image_mode
        self.model_complexity = model_complexity
        self.smooth_landmarks = smooth_landmarks
        self.enable_segmentation = enable_segmentation
        self.smooth_segmentation = smooth_segmentation
        self.detection_confidence = detection_confidence
        self.tracking_confidence = tracking_confidence

        # Initialize MediaPipe pose components
        self.mp_pose = mp.solutions.pose
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.enable_segmentation,
            smooth_segmentation=self.smooth_segmentation,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )

        # Store the previous landmarks for smoothing
        self.prev_landmarks = None
        self.smoothing_factor = 0.7  # Higher = more smoothing

    def detect_pose(self, frame, smooth=True):
        """Detect pose in the given frame.

        Args:
            frame: Image frame from a video source
            smooth: Whether to apply additional temporal smoothing

        Returns:
            The results from MediaPipe pose processing
        """
        # Convert the BGR image to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process the frame and find poses
        results = self.pose.process(frame_rgb)

        # Apply additional smoothing if enabled and landmarks detected
        if smooth and results.pose_landmarks and self.prev_landmarks:
            self._apply_smoothing(results)

        # Update previous landmarks
        if results.pose_landmarks:
            self.prev_landmarks = results.pose_landmarks

        return results

    def _apply_smoothing(self, results):
        """Apply additional temporal smoothing to landmarks.

        This applies exponential moving average smoothing to reduce jitter.

        Args:
            results: The MediaPipe pose results to smooth
        """
        if not self.prev_landmarks:
            return

        for i in range(len(results.pose_landmarks.landmark)):
            results.pose_landmarks.landmark[i].x = self.smoothing_factor * self.prev_landmarks.landmark[i].x + \
                                                 (1 - self.smoothing_factor) * results.pose_landmarks.landmark[i].x
            results.pose_landmarks.landmark[i].y = self.smoothing_factor * self.prev_landmarks.landmark[i].y + \
                                                 (1 - self.smoothing_factor) * results.pose_landmarks.landmark[i].y
            results.pose_landmarks.landmark[i].z = self.smoothing_factor * self.prev_landmarks.landmark[i].z + \
                                                 (1 - self.smoothing_factor) * results.pose_landmarks.landmark[i].z

    def landmark_to_pixel(self, landmark, img_width, img_height):
        """Convert normalized landmark to pixel coordinates.

        Args:
            landmark: MediaPipe normalized landmark (range [0.0, 1.0])
            img_width: Width of the image
            img_height: Height of the image

        Returns:
            Tuple (x, y) with pixel coordinates
        """
        x = int(landmark.x * img_width)
        y = int(landmark.y * img_height)
        return (x, y)

    def reset(self):
        """Reset the pose tracking by clearing previous landmarks."""
        self.prev_landmarks = None
        # Recreate the pose object to fully reset internal state
        self.pose = self.mp_pose.Pose(
            static_image_mode=self.static_image_mode,
            model_complexity=self.model_complexity,
            smooth_landmarks=self.smooth_landmarks,
            enable_segmentation=self.enable_segmentation,
            smooth_segmentation=self.smooth_segmentation,
            min_detection_confidence=self.detection_confidence,
            min_tracking_confidence=self.tracking_confidence
        )

    def get_pose_landmarks(self):
        """Get the landmark names and indices for reference.

        Returns:
            Dictionary mapping landmark names to their indices
        """
        return {
            "NOSE": 0,
            "LEFT_EYE_INNER": 1,
            "LEFT_EYE": 2,
            "LEFT_EYE_OUTER": 3,
            "RIGHT_EYE_INNER": 4,
            "RIGHT_EYE": 5,
            "RIGHT_EYE_OUTER": 6,
            "LEFT_EAR": 7,
            "RIGHT_EAR": 8,
            "MOUTH_LEFT": 9,
            "MOUTH_RIGHT": 10,
            "LEFT_SHOULDER": 11,
            "RIGHT_SHOULDER": 12,
            "LEFT_ELBOW": 13,
            "RIGHT_ELBOW": 14,
            "LEFT_WRIST": 15,
            "RIGHT_WRIST": 16,
            "LEFT_PINKY": 17,
            "RIGHT_PINKY": 18,
            "LEFT_INDEX": 19,
            "RIGHT_INDEX": 20,
            "LEFT_THUMB": 21,
            "RIGHT_THUMB": 22,
            "LEFT_HIP": 23,
            "RIGHT_HIP": 24,
            "LEFT_KNEE": 25,
            "RIGHT_KNEE": 26,
            "LEFT_ANKLE": 27,
            "RIGHT_ANKLE": 28,
            "LEFT_HEEL": 29,
            "RIGHT_HEEL": 30,
            "LEFT_FOOT_INDEX": 31,
            "RIGHT_FOOT_INDEX": 32
        }
