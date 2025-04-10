#!/usr/bin/env python3
"""
Camera to Robot Calibration Module

This module handles the calibration between the RealSense camera coordinate system
and the Dobot robot coordinate system. It calculates the transformation matrix
needed to convert camera coordinates to robot coordinates.

Calibration Process:
1. Define corresponding points in both camera and robot coordinate systems
2. Calculate the affine transformation matrix between these points
3. Save the transformation matrix for use in the main application

Usage:
    from calibration import CameraRobotCalibrator
    
    # Create calibrator
    calibrator = CameraRobotCalibrator()
    
    # Get transformation matrix
    transformation_matrix = calibrator.get_transformation_matrix()
    
    # Convert camera coordinates to robot coordinates
    camera_point = np.array([x, y, z, 1])
    robot_point = calibrator.transform_point(camera_point)
"""

import numpy as np
import cv2
import os
import json
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class CameraRobotCalibrator:
    """
    Handles calibration between camera and robot coordinate systems.
    
    This class calculates and stores the transformation matrix needed to convert
    coordinates from the camera's coordinate system to the robot's coordinate system.
    """
    
    def __init__(self, calibration_file=None):
        """
        Initialize the calibrator with optional calibration data.
        
        Args:
            calibration_file (str, optional): Path to a saved calibration file.
                                             If provided, loads the transformation matrix from this file.
        """
        # Define the calibration points in camera coordinates (in meters, converted to mm)
        self.camera_points = np.array([
            [-0.02 * 1000, 0.05 * 1000, 0.59 * 1000],  # Point 1
            [0.08 * 1000, .05 * 1000, 0.61 * 1000],    # Point 2
            [-.03 * 1000, 0.02 * 1000, 0.57 * 1000],   # Point 3
            [0.02 * 1000, 0.02 * 1000, 0.59 * 1000],   # Point 4
            [0.12 * 1000, -.01 * 1000, 0.62 * 1000]    # Point 5
        ], dtype=np.float32)
        
        # Define the corresponding points in robot coordinates (in mm)
        self.robot_points = np.array([
            [293.1320, -25.2514, 9.6383],   # Point 1
            [287.2104, 73.1777, 7.6396],    # Point 2
            [252.8839, -37.8998, 7.0930],   # Point 3
            [252.7630, 16.9580, 8.6920],    # Point 4
            [218.9429, 111.0479, 9.4959]    # Point 5
        ], dtype=np.float32)
        
        # Initialize transformation matrix
        self.transformation_matrix = None
        
        # Load calibration if file is provided
        if calibration_file and os.path.exists(calibration_file):
            self.load_calibration(calibration_file)
        else:
            # Calculate transformation matrix
            self.calculate_transformation()
    
    def calculate_transformation(self):
        """
        Calculate the transformation matrix between camera and robot coordinate systems.
        
        Returns:
            bool: True if transformation was calculated successfully, False otherwise.
        """
        try:
            # Calculate the affine transformation between camera and robot points
            retval, transformation_matrix, inliers = cv2.estimateAffine3D(
                self.camera_points, self.robot_points)
            
            if not retval:
                logger.error("Failed to estimate transformation matrix")
                return False
            
            self.transformation_matrix = transformation_matrix
            logger.info("Transformation matrix calculated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error calculating transformation: {e}")
            return False
    
    def transform_point(self, camera_point):
        """
        Transform a point from camera coordinates to robot coordinates.
        
        Args:
            camera_point (np.ndarray): Point in camera coordinates [x, y, z, 1]
            
        Returns:
            np.ndarray: Point in robot coordinates [x, y, z]
        """
        if self.transformation_matrix is None:
            logger.error("Transformation matrix not calculated")
            return None
        
        try:
            # Apply transformation
            robot_point = np.dot(self.transformation_matrix, camera_point.T)
            return robot_point[:3]  # Return only x, y, z coordinates
            
        except Exception as e:
            logger.error(f"Error transforming point: {e}")
            return None
    
    def save_calibration(self, file_path):
        """
        Save the calibration data to a file.
        
        Args:
            file_path (str): Path to save the calibration data
            
        Returns:
            bool: True if saved successfully, False otherwise
        """
        if self.transformation_matrix is None:
            logger.error("No transformation matrix to save")
            return False
        
        try:
            # Create directory if it doesn't exist
            if file_path and os.path.dirname(file_path):
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
            
            # Save transformation matrix
            np.save(file_path, self.transformation_matrix)
            logger.info(f"Calibration saved to {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving calibration: {e}")
            return False
    
    def load_calibration(self, file_path):
        """
        Load calibration data from a file.
        
        Args:
            file_path (str): Path to the calibration file
            
        Returns:
            bool: True if loaded successfully, False otherwise
        """
        if not os.path.exists(file_path):
            logger.error(f"Calibration file not found: {file_path}")
            return False
        
        try:
            # Load transformation matrix
            self.transformation_matrix = np.load(file_path)
            logger.info(f"Calibration loaded from {file_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading calibration: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Create calibrator
    calibrator = CameraRobotCalibrator()
    
    # Save calibration with a default path
    default_calibration_file = "calibration_data.npy"
    calibrator.save_calibration(default_calibration_file)
    
    # Example: transform a point
    camera_point = np.array([0.05 * 1000, 0.03 * 1000, 0.6 * 1000, 1])
    robot_point = calibrator.transform_point(camera_point)
    
    if robot_point is not None:
        print(f"Camera point: {camera_point[:3]}")
        print(f"Robot point: {robot_point}") 