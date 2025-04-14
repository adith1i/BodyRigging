import cv2
import time
import numpy as np


def calculate_fps(prev_time, frame_count, update_interval=1.0):
    """Calculate FPS over a time interval.

    Args:
        prev_time: Previous time measurement
        frame_count: Current frame count
        update_interval: How often to update FPS calculation (seconds)

    Returns:
        Tuple of (fps, new_prev_time, new_frame_count)
    """
    current_time = time.time()
    frame_count += 1

    # Calculate FPS every update_interval seconds
    time_diff = current_time - prev_time
    fps = 0

    if time_diff >= update_interval:
        fps = frame_count / time_diff
        prev_time = current_time
        frame_count = 0

    return fps, prev_time, frame_count


def draw_fps(image, fps):
    """Draw FPS information on the image.

    Args:
        image: The image to draw on
        fps: Current FPS value

    Returns:
        The image with FPS text
    """
    cv2.putText(image, f"FPS: {fps:.1f}", (10, 30),
              cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 4)
    cv2.putText(image, f"FPS: {fps:.1f}", (10, 30),
              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image


def calculate_distance(p1, p2):
    """Calculate Euclidean distance between two points.

    Args:
        p1: First point (x1, y1)
        p2: Second point (x2, y2)

    Returns:
        Euclidean distance
    """
    return np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)


def normalize_landmarks(landmarks, img_width, img_height):
    """Convert landmarks to pixel coordinates and normalize.

    Args:
        landmarks: MediaPipe landmarks
        img_width: Image width
        img_height: Image height

    Returns:
        List of normalized landmark coordinates as (x, y) tuples
    """
    normalized = []
    for landmark in landmarks:
        x = int(landmark.x * img_width)
        y = int(landmark.y * img_height)
        normalized.append((x, y))
    return normalized


def apply_one_euro_filter(x, prev_x=None, min_cutoff=1.0, beta=0.0):
    """Apply One Euro Filter for jitter reduction.

    Implementation of the One Euro Filter by Casiez et al. 2012
    Simple version without derivative component.

    Args:
        x: Current raw value
        prev_x: Previous filtered value
        min_cutoff: Minimum cutoff frequency
        beta: Cutoff slope

    Returns:
        Filtered value
    """
    if prev_x is None:
        return x

    # Alpha value for filter (smoothing factor)
    alpha = min_cutoff / (min_cutoff + beta)

    # Apply filter
    filtered_x = alpha * x + (1 - alpha) * prev_x
    return filtered_x


def create_transparent_overlay(image, mask, color=(0, 255, 0), alpha=0.3):
    """Create a transparent colored overlay using a mask.

    Args:
        image: Base image
        mask: Binary mask defining the overlay region
        color: RGB color for the overlay
        alpha: Transparency factor (0-1)

    Returns:
        Image with transparent overlay
    """
    # Ensure mask has proper dimensions
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, axis=2)
        mask = np.repeat(mask, 3, axis=2)

    # Create colored overlay
    overlay = np.zeros_like(image)
    overlay[:] = color

    # Apply transparent overlay where mask is true
    result = np.where(
        mask > 0,
        cv2.addWeighted(image, 1-alpha, overlay, alpha, 0),
        image
    )

    return result


def draw_bounding_box(image, landmarks, padding=20, color=(0, 255, 0), thickness=2):
    """Draw a bounding box around the detected pose.

    Args:
        image: Image to draw on
        landmarks: MediaPipe landmarks
        padding: Extra space around the landmarks
        color: RGB color for the bounding box
        thickness: Line thickness

    Returns:
        Image with bounding box
    """
    # Convert landmarks to pixel coordinates
    h, w, _ = image.shape
    points = []
    for lm in landmarks:
        points.append((int(lm.x * w), int(lm.y * h)))

    # Find bounding box coordinates
    x_coordinates = [p[0] for p in points]
    y_coordinates = [p[1] for p in points]

    x_min = max(0, min(x_coordinates) - padding)
    y_min = max(0, min(y_coordinates) - padding)
    x_max = min(w, max(x_coordinates) + padding)
    y_max = min(h, max(y_coordinates) + padding)

    # Draw bounding box
    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, thickness)

    return image
