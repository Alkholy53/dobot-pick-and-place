#!/usr/bin/env python3
"""
Dobot Magician Pick and Place with Computer Vision
This script implements object detection and robotic arm control using YOLOv10 and RealSense camera.
"""

import logging
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np
import cv2
import pyrealsense2 as rs
from ultralytics import YOLO
from Dobot import DoBotArm
from calibration import CameraRobotCalibrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Object classes
classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
              "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
              "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
              "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
              "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
              "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
              "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
              "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
              "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
              "teddy bear", "hair drier", "toothbrush"]

@dataclass
class CameraConfig:
    """Camera configuration parameters"""
    width: int = 640
    height: int = 480
    fps: int = 30
    color_format: int = rs.format.bgr8
    depth_format: int = rs.format.z16

@dataclass
class DetectionConfig:
    """Object detection configuration"""
    model_path: str = "yolov10n.pt"
    target_classes: List[str] = ("carrot", "scissors")
    confidence_threshold: float = 0.5
    output_dir: str = "detected_images"

class CameraManager:
    """Manages RealSense camera operations"""
    def __init__(self, config: CameraConfig):
        self.config = config
        self.pipeline = rs.pipeline()
        self.setup_camera()

    def setup_camera(self):
        """Initialize and configure the RealSense camera"""
        try:
            config = rs.config()
            config.enable_stream(rs.stream.color, self.config.width, self.config.height, 
                               self.config.color_format, self.config.fps)
            config.enable_stream(rs.stream.depth, self.config.width, self.config.height, 
                               self.config.depth_format, self.config.fps)
            self.pipeline.start(config)
            
            # Get camera parameters
            profile = self.pipeline.get_active_profile()
            self.depth_sensor = profile.get_device().first_depth_sensor()
            self.depth_scale = self.depth_sensor.get_depth_scale()
            color_stream = profile.get_stream(rs.stream.color)
            self.intrinsics = color_stream.as_video_stream_profile().get_intrinsics()
            
            logger.info("Camera setup completed successfully")
        except Exception as e:
            logger.error(f"Failed to setup camera: {e}")
            raise

    def get_frames(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Get color and depth frames from the camera"""
        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                return None, None

            return (np.asanyarray(color_frame.get_data()),
                   np.asanyarray(depth_frame.get_data()))
        except Exception as e:
            logger.error(f"Error getting frames: {e}")
            return None, None

    def cleanup(self):
        """Clean up camera resources"""
        self.pipeline.stop()

class ObjectDetector:
    """Handles object detection using YOLOv10"""
    def __init__(self, config: DetectionConfig):
        self.config = config
        self.model = YOLO(config.model_path)
        os.makedirs(config.output_dir, exist_ok=True)

    def process_frame(self, img: np.ndarray, depth_image: np.ndarray, 
                     intrinsics: rs.intrinsics, depth_scale: float) -> List[Tuple]:
        """Process a frame to detect objects and get 3D coordinates"""
        try:
            results = self.model(img, stream=True)
            coordinates = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    if box.conf[0] < self.config.confidence_threshold:
                        continue

                    # Get bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    # Get depth and 3D coordinates
                    depth = depth_image[cy, cx] * depth_scale
                    point = rs.rs2_deproject_pixel_to_point(intrinsics, [cx, cy], depth)
                    x, y, z = point

                    # Get class name
                    class_id = int(box.cls[0])
                    detected_class = classNames[class_id]

                    if detected_class in self.config.target_classes:
                        coordinates.append((detected_class, x, y, z))
                        self._draw_detection(img, x1, y1, x2, y2, cx, cy, detected_class, x, y, z)

            if coordinates:
                self._save_detection(img)

            return coordinates
        except Exception as e:
            logger.error(f"Error processing frame: {e}")
            return []

    def _draw_detection(self, img: np.ndarray, x1: int, y1: int, x2: int, y2: int,
                       cx: int, cy: int, class_name: str, x: float, y: float, z: float):
        """Draw detection results on the image"""
        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
        label = f"{class_name} ({x:.2f}, {y:.2f}, {z:.2f}m)"
        cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)

    def _save_detection(self, img: np.ndarray):
        """Save the detection image"""
        output_path = os.path.join(self.config.output_dir, "detected_image.jpg")
        cv2.imwrite(output_path, img)

class RobotController:
    """Controls the Dobot robotic arm"""
    def __init__(self, home_position: Tuple[float, float, float], calibration_file=None):
        self.dobot = DoBotArm(*home_position)
        
        # Initialize the calibrator
        self.calibrator = CameraRobotCalibrator(calibration_file)
        
        # Save calibration if it doesn't exist
        if calibration_file and not os.path.exists(calibration_file):
            self.calibrator.save_calibration(calibration_file)

    def move_to_object(self, x: float, y: float, z: float):
        """Move the robot to the detected object"""
        try:
            # Transform camera coordinates to robot coordinates
            camera_point = np.array([x * 1000, y * 1000, z * 1000, 1], dtype=np.float32)
            robot_point = self.calibrator.transform_point(camera_point)
            
            if robot_point is None:
                logger.error("Failed to transform coordinates")
                return
            
            rx, ry, rz = robot_point
            logger.info(f"Moving robot to coordinates: X={rx:.2f}, Y={ry:.2f}, Z={rz:.2f}")
            self.dobot.goPick2((rx, ry, rz), (267.2715, -94.2753, 14.4678))

        except Exception as e:
            logger.error(f"Error moving robot: {e}")

def main():
    """Main application entry point"""
    # Initialize configurations
    camera_config = CameraConfig()
    detection_config = DetectionConfig()
    
    # Calibration file path
    calibration_file = "calibration_data.npy"

    try:
        # Initialize components
        camera = CameraManager(camera_config)
        detector = ObjectDetector(detection_config)
        robot = RobotController((250, 0, 50), calibration_file)

        logger.info("Starting main loop...")
        while True:
            # Get camera frames
            color_frame, depth_frame = camera.get_frames()
            if color_frame is None or depth_frame is None:
                continue

            # Process frame and detect objects
            coordinates = detector.process_frame(
                color_frame, depth_frame, camera.intrinsics, camera.depth_scale)

            # Move robot to detected objects
            for class_name, x, y, z in coordinates:
                logger.info(f"Detected {class_name} at X: {x:.2f}, Y: {y:.2f}, Z: {z:.2f}m")
                robot.move_to_object(x, y, z)

    except KeyboardInterrupt:
        logger.info("Application stopped by user")
    except Exception as e:
        logger.error(f"Application error: {e}")
    finally:
        camera.cleanup()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
