import bpy
import numpy as np
import mathutils

def calculate_camera_intrinsics(camera, resolution_x, resolution_y):
    """Calculate camera intrinsics matrix"""
    # Get camera parameters
    focal_length = camera.data.lens  # mm
    sensor_width = camera.data.sensor_width  # mm
    
    # Calculate focal length in pixels
    fx = focal_length * (resolution_x / sensor_width)
    fy = focal_length * (resolution_y / sensor_width)
    
    # Calculate principal point (center of image)
    cx = resolution_x / 2
    cy = resolution_y / 2
    
    # Create intrinsics matrix
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    return K

def get_camera_pose(camera):
    """Get camera rotation and translation"""
    # Get camera matrix directly from matrix_world
    matrix = camera.matrix_world
    
    # Extract rotation and translation
    R = np.array(matrix.to_3x3())
    t = np.array(matrix.translation)
    
    return R, t

def calculate_relative_poses(camera_matrices):
    """Calculate relative poses between all views"""
    relative_poses = []
    
    for i in range(len(camera_matrices)):
        for j in range(len(camera_matrices)):
            if i != j:
                # Calculate relative transform
                relative_transform = np.linalg.inv(camera_matrices[i]) @ camera_matrices[j]
                
                # Add to list
                relative_poses.append({
                    "from_view": i,
                    "to_view": j,
                    "transform_matrix": relative_transform.tolist()
                })
    
    return relative_poses

def save_pose_data(pose_matrix, filepath):
    """Save pose matrix to file"""
    with open(filepath, 'w') as f:
        for row in pose_matrix:
            f.write(" ".join([str(v) for v in row]) + "\n") 