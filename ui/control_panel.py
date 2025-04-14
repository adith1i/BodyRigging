import cv2
import numpy as np


class ControlPanel:
    """Simple UI control panel for visualization settings."""

    def __init__(self, width=300, height=720):
        """Initialize the control panel.

        Args:
            width: Panel width
            height: Panel height
        """
        self.width = width
        self.height = height
        self.panel_color = (240, 240, 240)  # Light gray

        # Define slider parameters
        self.sliders = {
            "joint_size": {
                "min": 1,
                "max": 20,
                "value": 5,
                "label": "Joint Size"
            },
            "line_thickness": {
                "min": 1,
                "max": 10,
                "value": 2,
                "label": "Line Thickness"
            },
            "smoothing": {
                "min": 0,
                "max": 100,
                "value": 70,
                "label": "Smoothing"
            },
            "trail_length": {
                "min": 0,
                "max": 100,
                "value": 20,
                "label": "Trail Length"
            }
        }

        # Define button parameters
        self.buttons = {
            "skeleton": {
                "active": True,
                "label": "Skeleton View",
                "position": (50, 400),
                "size": (200, 50)
            },
            "mesh": {
                "active": False,
                "label": "Mesh View",
                "position": (50, 460),
                "size": (200, 50)
            },
            "contour": {
                "active": False,
                "label": "Contour View",
                "position": (50, 520),
                "size": (200, 50)
            },
            "reset": {
                "active": False,
                "label": "Reset Tracking",
                "position": (50, 580),
                "size": (200, 50)
            },
            "record": {
                "active": False,
                "label": "Record",
                "position": (50, 640),
                "size": (200, 50)
            }
        }

        # Track whether mouse is pressed
        self.mouse_pressed = False

        # Create window and set mouse callback
        cv2.namedWindow("Controls")
        cv2.setMouseCallback("Controls", self._mouse_callback)

    def show(self):
        """Create and display the control panel window.

        Returns:
            Control panel image
        """
        # Create panel background
        panel = np.ones((self.height, self.width, 3), dtype=np.uint8) * self.panel_color

        # Draw title
        cv2.putText(panel, "Body Rigging Controls", (20, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)

        # Draw sliders
        y_offset = 70
        for slider_id, slider_data in self.sliders.items():
            y_pos = y_offset
            # Draw slider label
            cv2.putText(panel, slider_data["label"], (20, y_pos),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1)

            # Draw slider track
            track_start = (20, y_pos + 20)
            track_end = (self.width - 20, y_pos + 20)
            cv2.line(panel, track_start, track_end, (100, 100, 100), 2)

            # Calculate slider position
            value_range = slider_data["max"] - slider_data["min"]
            position_x = int(track_start[0] + (track_end[0] - track_start[0]) *
                          (slider_data["value"] - slider_data["min"]) / value_range)

            # Draw slider handle
            cv2.circle(panel, (position_x, track_start[1]), 10, (50, 50, 200), -1)

            # Draw current value
            cv2.putText(panel, str(slider_data["value"]), (track_end[0] + 5, y_pos + 25),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            y_offset += 60

        # Draw buttons
        for button_id, button_data in self.buttons.items():
            position = button_data["position"]
            size = button_data["size"]
            label = button_data["label"]
            active = button_data["active"]

            # Draw button rectangle
            x, y = position
            w, h = size
            if active:
                # Active button color
                cv2.rectangle(panel, (x, y), (x + w, y + h), (0, 200, 0), -1)
                text_color = (255, 255, 255)
            else:
                # Inactive button color
                cv2.rectangle(panel, (x, y), (x + w, y + h), (200, 200, 200), -1)
                text_color = (0, 0, 0)

            # Draw button border
            cv2.rectangle(panel, (x, y), (x + w, y + h), (100, 100, 100), 1)

            # Draw button label (centered)
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            cv2.putText(panel, label,
                      (x + (w - label_size[0]) // 2, y + (h + label_size[1]) // 2),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 1)

        # Display the panel
        cv2.imshow("Controls", panel)
        cv2.waitKey(1)

        return panel

    def get_parameters(self):
        """Get the current parameter values.

        Returns:
            Dictionary with current parameter values
        """
        params = {}

        # Get slider values
        for slider_id, slider_data in self.sliders.items():
            params[slider_id] = slider_data["value"]

        # Get button states
        for button_id, button_data in self.buttons.items():
            params[button_id] = button_data["active"]

        return params

    def _mouse_callback(self, event, x, y, flags, param):
        """Handle mouse events for sliders and buttons.

        Args:
            event: Mouse event type
            x, y: Mouse coordinates
            flags: Additional flags
            param: Additional parameters
        """
        # Mouse button down
        if event == cv2.EVENT_LBUTTONDOWN:
            self.mouse_pressed = True
            self._handle_click(x, y)

        # Mouse move with button pressed (for sliders)
        elif event == cv2.EVENT_MOUSEMOVE and self.mouse_pressed:
            self._handle_drag(x, y)

        # Mouse button up
        elif event == cv2.EVENT_LBUTTONUP:
            self.mouse_pressed = False

    def _handle_click(self, x, y):
        """Handle mouse click for buttons and sliders.

        Args:
            x, y: Mouse coordinates
        """
        # Check if clicking on a button
        for button_id, button_data in self.buttons.items():
            bx, by = button_data["position"]
            bw, bh = button_data["size"]

            if bx <= x <= bx + bw and by <= y <= by + bh:
                # For view toggle buttons, only one can be active at a time
                if button_id in ["skeleton", "mesh", "contour"]:
                    for view_id in ["skeleton", "mesh", "contour"]:
                        self.buttons[view_id]["active"] = (view_id == button_id)
                else:
                    # For other buttons, toggle active state
                    self.buttons[button_id]["active"] = not button_data["active"]
                return

        # Check if clicking on a slider
        y_offset = 70
        for slider_id, slider_data in self.sliders.items():
            slider_y = y_offset + 20  # Vertical position of slider

            # Check if click is close to the slider
            if abs(y - slider_y) < 15 and 20 <= x <= self.width - 20:
                # Update slider position
                self._update_slider_value(slider_id, x)
                return

            y_offset += 60

    def _handle_drag(self, x, y):
        """Handle mouse drag for sliders.

        Args:
            x, y: Mouse coordinates
        """
        # Only check sliders
        y_offset = 70
        for slider_id, slider_data in self.sliders.items():
            slider_y = y_offset + 20  # Vertical position of slider

            # Check if drag is close to the slider
            if abs(y - slider_y) < 30 and 20 <= x <= self.width - 20:
                # Update slider position
                self._update_slider_value(slider_id, x)
                return

            y_offset += 60

    def _update_slider_value(self, slider_id, x):
        """Update slider value based on x position.

        Args:
            slider_id: ID of slider to update
            x: Horizontal mouse position
        """
        # Calculate the value based on position
        track_start = 20
        track_end = self.width - 20
        position_ratio = (x - track_start) / (track_end - track_start)

        slider = self.sliders[slider_id]
        value_range = slider["max"] - slider["min"]
        new_value = int(slider["min"] + position_ratio * value_range)

        # Clamp to valid range
        new_value = max(slider["min"], min(new_value, slider["max"]))

        # Update slider value
        self.sliders[slider_id]["value"] = new_value
