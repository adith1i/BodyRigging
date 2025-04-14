import numpy as np
from collections import deque
import math


class LandmarkSmoother:
    """Provides various smoothing algorithms for landmark data."""

    def __init__(self, landmarks_count=33):
        """Initialize the landmark smoother.

        Args:
            landmarks_count: Number of landmarks in the pose tracking
        """
        self.landmarks_count = landmarks_count

        # Initialize smoothing parameters
        self.smoothing_method = "EMA"  # Default method

        # EMA (Exponential Moving Average) parameters
        self.ema_alpha = 0.3  # Lower value = more smoothing
        self.prev_landmarks = None

        # One Euro Filter parameters
        self.one_euro_min_cutoff = 1.0
        self.one_euro_beta = 0.007
        self.one_euro_d_cutoff = 1.0
        self.one_euro_prev_filtered_val = None
        self.one_euro_prev_timestamp = None
        self.one_euro_prev_dx = None

        # Kalman filter parameters
        self.kalman_process_noise = 0.03
        self.kalman_measurement_noise = 0.1
        self.kalman_state = None
        self.kalman_covariance = None

        # Moving average window
        self.window_size = 5
        self.landmark_history = [deque(maxlen=self.window_size)
                               for _ in range(self.landmarks_count * 3)]  # x, y, z for each landmark

    def set_smoothing_method(self, method):
        """Set the smoothing method to use.

        Args:
            method: String name of the method ('EMA', 'ONE_EURO', 'KALMAN', 'MOVING_AVG')
        """
        valid_methods = ['EMA', 'ONE_EURO', 'KALMAN', 'MOVING_AVG']
        if method in valid_methods:
            self.smoothing_method = method
            self.reset()

    def smooth_landmarks(self, landmarks, timestamp=None):
        """Apply smoothing to landmarks based on chosen method.

        Args:
            landmarks: MediaPipe landmarks
            timestamp: Current timestamp (required for some filters)

        Returns:
            Smoothed landmarks
        """
        if landmarks is None:
            return None

        # Call the appropriate smoothing method
        if self.smoothing_method == "EMA":
            return self._apply_ema_smoothing(landmarks)
        elif self.smoothing_method == "ONE_EURO":
            return self._apply_one_euro_filter(landmarks, timestamp)
        elif self.smoothing_method == "KALMAN":
            return self._apply_kalman_filter(landmarks)
        elif self.smoothing_method == "MOVING_AVG":
            return self._apply_moving_average(landmarks)
        else:
            return landmarks  # No smoothing

    def reset(self):
        """Reset all smoothing states."""
        self.prev_landmarks = None
        self.one_euro_prev_filtered_val = None
        self.one_euro_prev_timestamp = None
        self.one_euro_prev_dx = None
        self.kalman_state = None
        self.kalman_covariance = None
        self.landmark_history = [deque(maxlen=self.window_size)
                               for _ in range(self.landmarks_count * 3)]

    def _apply_ema_smoothing(self, landmarks):
        """Apply Exponential Moving Average smoothing.

        Args:
            landmarks: MediaPipe landmarks

        Returns:
            Smoothed landmarks
        """
        # If first frame, initialize history
        if self.prev_landmarks is None:
            self.prev_landmarks = landmarks
            return landmarks

        # Apply EMA smoothing to each landmark
        smoothed_landmarks = []
        for i, landmark in enumerate(landmarks.landmark):
            # Check if we have previous data for this landmark
            if i < len(self.prev_landmarks.landmark):
                prev = self.prev_landmarks.landmark[i]

                # Apply EMA formula: new_value = alpha * current + (1 - alpha) * prev
                smoothed_x = self.ema_alpha * landmark.x + (1 - self.ema_alpha) * prev.x
                smoothed_y = self.ema_alpha * landmark.y + (1 - self.ema_alpha) * prev.y
                smoothed_z = self.ema_alpha * landmark.z + (1 - self.ema_alpha) * prev.z
                smoothed_visibility = landmark.visibility  # Pass through visibility

                # Create a new landmark with smoothed values
                smooth_landmark = type(landmark)()
                smooth_landmark.x = smoothed_x
                smooth_landmark.y = smoothed_y
                smooth_landmark.z = smoothed_z
                smooth_landmark.visibility = smoothed_visibility

                smoothed_landmarks.append(smooth_landmark)
            else:
                # If no previous data, use raw landmark
                smoothed_landmarks.append(landmark)

        # Create a new landmark collection with updated landmarks
        result = type(landmarks)()
        result.landmark.extend(smoothed_landmarks)

        # Update previous landmarks
        self.prev_landmarks = result

        return result

    def _apply_one_euro_filter(self, landmarks, timestamp):
        """Apply One Euro Filter for smoother yet responsive filtering.

        This is an implementation of the One Euro Filter by Casiez et al.

        Args:
            landmarks: MediaPipe landmarks
            timestamp: Current time in seconds

        Returns:
            Smoothed landmarks
        """
        # If timestamp not provided, use a constant increment
        if timestamp is None:
            if self.one_euro_prev_timestamp is None:
                self.one_euro_prev_timestamp = 0
            timestamp = self.one_euro_prev_timestamp + 0.03  # Assume 30fps

        # If first frame, initialize history
        if self.one_euro_prev_filtered_val is None:
            self.one_euro_prev_filtered_val = [
                (lm.x, lm.y, lm.z) for lm in landmarks.landmark
            ]
            self.one_euro_prev_timestamp = timestamp
            self.one_euro_prev_dx = [(0, 0, 0) for _ in landmarks.landmark]
            return landmarks

        # Calculate time difference
        dt = timestamp - self.one_euro_prev_timestamp
        if dt <= 0:
            dt = 0.03  # Assume 30fps if timestamps are invalid

        smoothed_landmarks = []
        for i, landmark in enumerate(landmarks.landmark):
            if i < len(self.one_euro_prev_filtered_val):
                current_val = (landmark.x, landmark.y, landmark.z)
                prev_val = self.one_euro_prev_filtered_val[i]
                prev_dx = self.one_euro_prev_dx[i]

                # Calculate derivative (rate of change)
                dx = tuple((curr - prev) / dt for curr, prev in zip(current_val, prev_val))

                # Filter the derivative
                edx = tuple(self._one_euro_alpha(self.one_euro_d_cutoff, dt) * dx_val +
                          (1 - self._one_euro_alpha(self.one_euro_d_cutoff, dt)) * prev_dx_val
                          for dx_val, prev_dx_val in zip(dx, prev_dx))

                # Use filtered derivative to adjust cutoff frequency
                cutoff = tuple(self.one_euro_min_cutoff + self.one_euro_beta * abs(edx_val)
                             for edx_val in edx)

                # Filter the signal
                filtered_val = tuple(self._one_euro_alpha(cutoff_val, dt) * curr_val +
                                   (1 - self._one_euro_alpha(cutoff_val, dt)) * prev_val
                                   for cutoff_val, curr_val, prev_val in zip(cutoff, current_val, prev_val))

                # Create smoothed landmark
                smooth_landmark = type(landmark)()
                smooth_landmark.x = filtered_val[0]
                smooth_landmark.y = filtered_val[1]
                smooth_landmark.z = filtered_val[2]
                smooth_landmark.visibility = landmark.visibility

                smoothed_landmarks.append(smooth_landmark)

                # Update state for next iteration
                self.one_euro_prev_filtered_val[i] = filtered_val
                self.one_euro_prev_dx[i] = edx
            else:
                # If no previous data, use raw landmark
                smoothed_landmarks.append(landmark)
                # Initialize state for this new landmark
                if len(self.one_euro_prev_filtered_val) <= i:
                    self.one_euro_prev_filtered_val.append((landmark.x, landmark.y, landmark.z))
                    self.one_euro_prev_dx.append((0, 0, 0))

        # Update timestamp
        self.one_euro_prev_timestamp = timestamp

        # Create a new landmark collection with updated landmarks
        result = type(landmarks)()
        result.landmark.extend(smoothed_landmarks)

        return result

    def _one_euro_alpha(self, cutoff, dt):
        """Calculate alpha value for One Euro Filter.

        Args:
            cutoff: Cutoff frequency
            dt: Time delta

        Returns:
            Alpha value for filter
        """
        r = 2 * math.pi * cutoff * dt
        return r / (r + 1)

    def _apply_kalman_filter(self, landmarks):
        """Apply Kalman filter for smooth, predictive tracking.

        Args:
            landmarks: MediaPipe landmarks

        Returns:
            Smoothed landmarks
        """
        # If first frame, initialize Kalman state
        if self.kalman_state is None:
            state_dim = self.landmarks_count * 3 * 2  # position and velocity for x, y, z of each landmark
            self.kalman_state = np.zeros(state_dim)

            # Initialize positions from first frame
            for i, landmark in enumerate(landmarks.landmark):
                self.kalman_state[i*6] = landmark.x      # x position
                self.kalman_state[i*6+1] = 0             # x velocity
                self.kalman_state[i*6+2] = landmark.y    # y position
                self.kalman_state[i*6+3] = 0             # y velocity
                self.kalman_state[i*6+4] = landmark.z    # z position
                self.kalman_state[i*6+5] = 0             # z velocity

            # Initialize covariance matrix
            self.kalman_covariance = np.eye(state_dim) * 0.1

            return landmarks

        # 1. Prediction step
        # Transition matrix (constant velocity model)
        dt = 1.0  # Assuming constant time step
        F = np.eye(len(self.kalman_state))

        # Update position with velocity
        for i in range(self.landmarks_count):
            base_idx = i * 6
            F[base_idx, base_idx+1] = dt    # x position += x velocity * dt
            F[base_idx+2, base_idx+3] = dt  # y position += y velocity * dt
            F[base_idx+4, base_idx+5] = dt  # z position += z velocity * dt

        # Predict state
        predicted_state = F @ self.kalman_state

        # Predict covariance
        Q = np.eye(len(self.kalman_state)) * self.kalman_process_noise
        predicted_covariance = F @ self.kalman_covariance @ F.T + Q

        # 2. Update step
        # Measurement matrix (we only measure position, not velocity)
        H = np.zeros((self.landmarks_count * 3, len(self.kalman_state)))
        for i in range(self.landmarks_count):
            H[i*3, i*6] = 1     # x position
            H[i*3+1, i*6+2] = 1  # y position
            H[i*3+2, i*6+4] = 1  # z position

        # Measurement from current landmarks
        measurement = np.zeros(self.landmarks_count * 3)
        for i, landmark in enumerate(landmarks.landmark):
            measurement[i*3] = landmark.x
            measurement[i*3+1] = landmark.y
            measurement[i*3+2] = landmark.z

        # Calculate Kalman gain
        R = np.eye(len(measurement)) * self.kalman_measurement_noise
        S = H @ predicted_covariance @ H.T + R
        K = predicted_covariance @ H.T @ np.linalg.inv(S)

        # Update state
        innovation = measurement - H @ predicted_state
        updated_state = predicted_state + K @ innovation

        # Update covariance
        updated_covariance = (np.eye(len(self.kalman_state)) - K @ H) @ predicted_covariance

        # Save updated state and covariance
        self.kalman_state = updated_state
        self.kalman_covariance = updated_covariance

        # Create smoothed landmarks from filtered state
        smoothed_landmarks = []
        for i, landmark in enumerate(landmarks.landmark):
            smooth_landmark = type(landmark)()
            smooth_landmark.x = self.kalman_state[i*6]      # x position
            smooth_landmark.y = self.kalman_state[i*6+2]    # y position
            smooth_landmark.z = self.kalman_state[i*6+4]    # z position
            smooth_landmark.visibility = landmark.visibility
            smoothed_landmarks.append(smooth_landmark)

        # Create a new landmark collection with updated landmarks
        result = type(landmarks)()
        result.landmark.extend(smoothed_landmarks)

        return result

    def _apply_moving_average(self, landmarks):
        """Apply simple moving average filter.

        Args:
            landmarks: MediaPipe landmarks

        Returns:
            Smoothed landmarks
        """
        # Update history with current landmarks
        for i, landmark in enumerate(landmarks.landmark):
            base_idx = i * 3
            self.landmark_history[base_idx].append(landmark.x)
            self.landmark_history[base_idx+1].append(landmark.y)
            self.landmark_history[base_idx+2].append(landmark.z)

        # Create smoothed landmarks using average of history
        smoothed_landmarks = []
        for i, landmark in enumerate(landmarks.landmark):
            base_idx = i * 3
            x_history = self.landmark_history[base_idx]
            y_history = self.landmark_history[base_idx+1]
            z_history = self.landmark_history[base_idx+2]

            # Skip if not enough history
            if len(x_history) == 0:
                smoothed_landmarks.append(landmark)
                continue

            # Calculate averages
            avg_x = sum(x_history) / len(x_history)
            avg_y = sum(y_history) / len(y_history)
            avg_z = sum(z_history) / len(z_history)

            # Create smoothed landmark
            smooth_landmark = type(landmark)()
            smooth_landmark.x = avg_x
            smooth_landmark.y = avg_y
            smooth_landmark.z = avg_z
            smooth_landmark.visibility = landmark.visibility

            smoothed_landmarks.append(smooth_landmark)

        # Create a new landmark collection with updated landmarks
        result = type(landmarks)()
        result.landmark.extend(smoothed_landmarks)

        return result
