import cv2
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import os


class DataDisplay:
    """Handles visualization of motion data and analytics."""

    def __init__(self, history_length=100):
        """Initialize the data display.

        Args:
            history_length: Number of frames to keep in history
        """
        self.history_length = history_length

        # Data storage for different metrics
        self.angle_history = {
            'elbow_left': deque(maxlen=history_length),
            'elbow_right': deque(maxlen=history_length),
            'knee_left': deque(maxlen=history_length),
            'knee_right': deque(maxlen=history_length)
        }

        self.speed_history = {
            'wrist_left': deque(maxlen=history_length),
            'wrist_right': deque(maxlen=history_length),
            'ankle_left': deque(maxlen=history_length),
            'ankle_right': deque(maxlen=history_length)
        }

        self.recording = False
        self.recording_start_time = 0
        self.recording_data = []
        self.output_dir = "recordings"

        # Create recording directory if it doesn't exist
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

    def update(self, motion_analyzer):
        """Update data from the motion analyzer.

        Args:
            motion_analyzer: MotionAnalyzer instance with current data
        """
        # Update angle history
        for joint_name in self.angle_history.keys():
            angle = motion_analyzer.get_joint_angle(joint_name)
            if angle is not None:
                self.angle_history[joint_name].append(angle)
            elif len(self.angle_history[joint_name]) > 0:
                # If no new data, repeat the last value
                self.angle_history[joint_name].append(self.angle_history[joint_name][-1])

        # Update speed history
        for joint_name in self.speed_history.keys():
            speed = motion_analyzer.get_speed(joint_name)
            if speed is not None:
                self.speed_history[joint_name].append(speed)
            elif len(self.speed_history[joint_name]) > 0:
                # If no new data, repeat the last value
                self.speed_history[joint_name].append(0)

        # Record data if recording is active
        if self.recording:
            timestamp = time.time() - self.recording_start_time
            frame_data = {
                'timestamp': timestamp,
                'angles': {},
                'speeds': {}
            }

            # Record angles
            for joint_name in self.angle_history.keys():
                if len(self.angle_history[joint_name]) > 0:
                    frame_data['angles'][joint_name] = self.angle_history[joint_name][-1]

            # Record speeds
            for joint_name in self.speed_history.keys():
                if len(self.speed_history[joint_name]) > 0:
                    frame_data['speeds'][joint_name] = self.speed_history[joint_name][-1]

            self.recording_data.append(frame_data)

    def start_recording(self):
        """Start recording motion data."""
        self.recording = True
        self.recording_start_time = time.time()
        self.recording_data = []
        print("Recording started.")

    def stop_recording(self):
        """Stop recording and save the data."""
        if not self.recording:
            return

        self.recording = False

        # Generate filename with timestamp
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"motion_data_{timestamp}"

        # Save data as CSV
        self._save_csv(filename)

        # Generate and save visualization
        self._generate_visualization(filename)

        print(f"Recording saved to {self.output_dir}/{filename}")

    def _save_csv(self, filename):
        """Save recorded data as CSV file.

        Args:
            filename: Base filename without extension
        """
        if not self.recording_data:
            return

        filepath = os.path.join(self.output_dir, f"{filename}.csv")

        # Determine all column names from the first frame
        angle_cols = list(self.recording_data[0]['angles'].keys())
        speed_cols = list(self.recording_data[0]['speeds'].keys())

        # Write header and data
        with open(filepath, 'w') as f:
            # Write header
            header = ['timestamp'] + [f"angle_{col}" for col in angle_cols] + [f"speed_{col}" for col in speed_cols]
            f.write(','.join(header) + '\n')

            # Write data rows
            for frame in self.recording_data:
                row = [str(frame['timestamp'])]

                # Add angle values
                for col in angle_cols:
                    row.append(str(frame['angles'].get(col, '')))

                # Add speed values
                for col in speed_cols:
                    row.append(str(frame['speeds'].get(col, '')))

                f.write(','.join(row) + '\n')

    def _generate_visualization(self, filename):
        """Generate visualization plots from recorded data.

        Args:
            filename: Base filename without extension
        """
        if not self.recording_data:
            return

        # Extract timestamps
        timestamps = [frame['timestamp'] for frame in self.recording_data]

        # Create figure with multiple subplots
        plt.figure(figsize=(12, 10))

        # Plot angles
        plt.subplot(2, 1, 1)
        for joint_name in self.angle_history.keys():
            values = [frame['angles'].get(joint_name, float('nan')) for frame in self.recording_data]
            plt.plot(timestamps, values, label=f"{joint_name}")

        plt.title('Joint Angles')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Angle (degrees)')
        plt.legend()
        plt.grid(True)

        # Plot speeds
        plt.subplot(2, 1, 2)
        for joint_name in self.speed_history.keys():
            values = [frame['speeds'].get(joint_name, float('nan')) for frame in self.recording_data]
            plt.plot(timestamps, values, label=f"{joint_name}")

        plt.title('Movement Speed')
        plt.xlabel('Time (seconds)')
        plt.ylabel('Speed (pixels/second)')
        plt.legend()
        plt.grid(True)

        plt.tight_layout()

        # Save figure
        plt.savefig(os.path.join(self.output_dir, f"{filename}.png"))
        plt.close()

    def create_data_view(self, width=640, height=480):
        """Create a real-time data visualization view.

        Args:
            width: View width
            height: View height

        Returns:
            Image with data visualization
        """
        # Create blank image
        image = np.ones((height, width, 3), dtype=np.uint8) * 255

        # Draw title
        cv2.putText(image, "Motion Analysis", (20, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Add recording indicator if active
        if self.recording:
            recording_time = time.time() - self.recording_start_time
            cv2.putText(image, f"Recording: {recording_time:.1f}s", (width - 200, 30),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        # Draw angle graph
        self._draw_graph(image,
                      title="Joint Angles (degrees)",
                      data=self.angle_history,
                      colors=[(255, 0, 0), (0, 0, 255), (255, 0, 255), (0, 255, 255)],
                      rect=(20, 50, width - 40, 180),
                      y_min=0, y_max=180)

        # Draw speed graph
        self._draw_graph(image,
                      title="Movement Speed (pixels/s)",
                      data=self.speed_history,
                      colors=[(0, 255, 0), (0, 128, 255), (128, 0, 255), (255, 255, 0)],
                      rect=(20, 280, width - 40, 180),
                      y_min=0, y_max=300)

        return image

    def _draw_graph(self, image, title, data, colors, rect, y_min, y_max):
        """Draw a simple graph on the image.

        Args:
            image: Image to draw on
            title: Graph title
            data: Dictionary of data series
            colors: List of colors for each series
            rect: (x, y, width, height) rectangle to draw in
            y_min, y_max: Value range for y-axis
        """
        x, y, w, h = rect

        # Draw background and border
        cv2.rectangle(image, (x, y), (x + w, y + h), (240, 240, 240), -1)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 0), 1)

        # Draw title
        cv2.putText(image, title, (x, y - 5),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

        # Draw y-axis labels
        cv2.putText(image, str(y_max), (x - 5, y + 15),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        cv2.putText(image, str(y_min), (x - 5, y + h - 5),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

        # Draw each data series
        for i, (series_name, series_data) in enumerate(data.items()):
            if not series_data:
                continue

            # Draw series name in legend
            legend_x = x + 10 + (i * 120)
            legend_y = y + 20
            cv2.putText(image, series_name, (legend_x + 15, legend_y),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.line(image, (legend_x, legend_y - 5), (legend_x + 10, legend_y - 5),
                   colors[i], 2)

            # Draw the data series
            points = []
            for j, value in enumerate(series_data):
                # Scale j to x coordinate
                point_x = x + int(j * w / self.history_length)

                # Scale value to y coordinate
                normalized_value = (value - y_min) / (y_max - y_min)
                point_y = y + h - int(normalized_value * h)

                # Clamp point within graph area
                point_y = max(y, min(y + h, point_y))

                points.append((point_x, point_y))

            # Draw lines between points
            if len(points) > 1:
                for j in range(1, len(points)):
                    cv2.line(image, points[j-1], points[j], colors[i], 2)
