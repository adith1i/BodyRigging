# Full Body Rigging and Tracking Application

A real-time full body tracking application that uses computer vision to create a digital skeleton that follows a user's movements through a webcam. The system provides fluid, responsive, and accurate body tracking suitable for applications in animation, fitness tracking, motion capture, and interactive experiences.

![Body Tracking Demo](https://via.placeholder.com/800x400?text=Body+Tracking+Demo)

## Features

### Core Functionality
- Full body pose estimation using MediaPipe (33 landmarks)
- Real-time skeletal visualization with customizable appearance
- Joint angle measurements and tracking
- Motion trails for analyzing movement paths
- 2D/3D visualization options

### Advanced Features
- Multiple smoothing algorithms for fluid motion:
  - One Euro filter for responsive yet smooth tracking
  - Kalman filtering for predictive motion estimation
  - Savitzky-Golay filter for preserving high-frequency components
  - Exponential moving average for general smoothing
  - Adaptive filtering based on movement velocity
- Performance optimization with multi-threading
- Dynamic resolution scaling for different hardware capabilities
- Body segmentation visualization
- Motion analysis and posture feedback

## Installation

### Prerequisites
- Python 3.6 or higher
- Webcam or video input device

### Steps
1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Running the Application
Run the main application:
```bash
python main.py
```

Or try the demo script:
```bash
python run_demo.py
```

### Controls
- **ESC**: Exit the application
- **Space**: Toggle between visualization modes
- **R**: Reset tracking
- **S**: Toggle between smoothing algorithms
- **P**: Pause/Resume tracking
- **V**: Toggle 2D/3D visualization
- **T**: Toggle motion trails
- **A**: Toggle angle measurements
- **F**: Toggle FPS display
- **+/-**: Adjust visualization size

## Project Structure

- `main.py`: Application entry point and main loop
- `pose_detector.py`: Body detection and pose estimation using MediaPipe
- `skeleton_visualizer.py`: Skeleton and joint visualization components
- `motion_analyzer.py`: Movement analysis, angle calculations, and posture feedback
- `smoothing.py`: Various motion smoothing algorithms implementation
- `utils.py`: Utility functions and helpers
- `ui/`: User interface components for controls and visualization
- `run_demo.py`: Demonstration script with pre-configured settings

## Technical Details

### Body Detection and Pose Estimation
MediaPipe's pose solution identifies 33 landmarks on the human body:
- Face and head: 5 points
- Torso: 4 points
- Arms: 8 points (4 per arm)
- Hands: 4 points (2 per hand)
- Legs: 8 points (4 per leg)
- Feet: 4 points (2 per foot)

### Performance Optimization
- Multi-threading to separate detection and visualization
- Temporal filtering for smooth motion
- Frame rate monitoring and adjustment
- Resolution scaling for different processing needs
- Configurable detail levels for different performance targets

### Visualization Options
- Skeletal structure with lines connecting anatomical landmarks
- Joint markers at key points with customizable shapes and sizes
- Body segmentation with optional mesh or contour visualization
- 3D pose estimation and visualization
- Motion trails for analyzing movement paths
- Angle measurements between body segments

## Applications

- Motion capture for animation and game development
- Fitness tracking and exercise analysis
- Posture correction and ergonomic assessment
- Interactive installations and digital experiences
- Physical therapy and rehabilitation monitoring

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## References
- [MediaPipe Pose Documentation](https://google.github.io/mediapipe/solutions/pose.html)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Human Pose Estimation Overview](https://medium.com/towards-artificial-intelligence/full-body-pose-tracking-ai-for-everyone-try-it-yourself-49c8366d8da1)
- [Body Landmark Model](https://google.github.io/mediapipe/images/mobile/pose_tracking_full_body_landmarks.png)
