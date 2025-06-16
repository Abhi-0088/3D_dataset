import bpy
import math
import os
import mathutils
import numpy as np

def normalize_model(obj):
    """
    Normalize the model by:
    1. Centering at origin
    2. Scaling to fit in unit sphere
    3. Aligning to principal axes
    """
    print("\n=== Starting Normalization ===")
    
    # Make sure the object is active and selected
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    
    # Get the object's bounding box in world space
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    
    # Calculate center in world space
    center = sum((mathutils.Vector(corner) for corner in bbox_corners), mathutils.Vector()) / 8
    print(f"Initial center: {center}")
    
    # First, move to origin
    obj.location = -center
    bpy.ops.object.transform_apply(location=True)
    print(f"After centering: {obj.location}")
    
    # Get vertices in world space for PCA
    mesh = obj.data
    # Get the evaluated mesh to account for modifiers
    depsgraph = bpy.context.evaluated_depsgraph_get()
    obj_eval = obj.evaluated_get(depsgraph)
    mesh_eval = obj_eval.to_mesh()
    
    # Get vertices from evaluated mesh
    vertices = np.array([obj.matrix_world @ v.co for v in mesh_eval.vertices])
    
    # Center vertices
    vertices_centered = vertices - np.mean(vertices, axis=0)
    
    try:
        # Ensure we have enough points for SVD (at least 3)
        if len(vertices_centered) < 3:
            raise np.linalg.LinAlgError("Not enough points for SVD")
            
        # Compute covariance matrix first
        cov_matrix = np.cov(vertices_centered.T)
        
        # Compute SVD of covariance matrix (which is always square)
        U, S, Vt = np.linalg.svd(cov_matrix, full_matrices=False, compute_uv=True)
        
        # Get rotation from Vt
        rotation = Vt.T
        
        # Ensure we have a valid rotation matrix
        if np.linalg.det(rotation) < 0:
            rotation[:, -1] *= -1  # Flip last axis for right-handed basis
            
        # Verify the rotation matrix is valid
        if not np.all(np.isfinite(rotation)):
            raise np.linalg.LinAlgError("Invalid rotation matrix computed")
            
    except np.linalg.LinAlgError as e:
        print(f"SVD computation failed: {str(e)}, falling back to eigenvalue decomposition")
        # Fallback to eigenvalue decomposition with better numerical stability
        cov_matrix = cov_matrix + np.eye(3) * np.finfo(float).eps  # Add small epsilon
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        idx = eigenvalues.argsort()[::-1]
        rotation = eigenvectors[:, idx]
    
    # Create rotation matrix - convert numpy array to list of lists for mathutils.Matrix
    rotation_matrix = mathutils.Matrix(rotation.tolist())
    
    # Apply rotation using matrix multiplication
    obj.matrix_world = rotation_matrix.to_4x4() @ obj.matrix_world
    bpy.ops.object.transform_apply(rotation=True)
    print(f"After rotation: {obj.rotation_euler}")
    
    # Clean up evaluated mesh
    obj_eval.to_mesh_clear()
    
    # Recalculate bounding box after rotation
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    
    # Calculate scale to fit in unit sphere
    max_dim = max((corner).length for corner in bbox_corners)
    # Scale to 0.8 units (80% of unit sphere) to leave some margin
    scale_factor = 0.8 / max_dim if max_dim > 0 else 1.0
    print(f"Calculated scale factor: {scale_factor}")
    
    # Apply scale using matrix multiplication
    scale_matrix = mathutils.Matrix.Scale(scale_factor, 4)
    obj.matrix_world = scale_matrix @ obj.matrix_world
    bpy.ops.object.transform_apply(scale=True)
    print(f"After scaling: {obj.scale}")
    
    # Verify final position
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    final_center = sum((mathutils.Vector(corner) for corner in bbox_corners), mathutils.Vector()) / 8
    print(f"Final center: {final_center}")
    
    # If not centered, apply one final centering
    if final_center.length > 0.001:  # Small threshold for floating point errors
        print("Applying final centering")
        obj.location = -final_center
        bpy.ops.object.transform_apply(location=True)
        print(f"Final location after centering: {obj.location}")
    
    # Final verification
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    xs = [v[0] for v in bbox_corners]
    ys = [v[1] for v in bbox_corners]
    zs = [v[2] for v in bbox_corners]
    print(f"Final bounding box dimensions:")
    print(f"X: {max(xs) - min(xs):.3f}")
    print(f"Y: {max(ys) - min(ys):.3f}")
    print(f"Z: {max(zs) - min(zs):.3f}")
    
    print("=== Normalization Complete ===\n")
    return obj

def import_3d_model(filepath):
    """Import 3D model based on file extension"""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model file not found: {filepath}")
        
    file_ext = os.path.splitext(filepath)[1].lower()
    
    try:
        if file_ext == '.obj':
            bpy.ops.import_scene.obj(filepath=filepath)
        elif file_ext in ['.glb', '.gltf']:
            # Clear existing objects
            bpy.ops.object.select_all(action='SELECT')
            bpy.ops.object.delete()
            
            # Import GLB/GLTF with specific settings
            bpy.ops.import_scene.gltf(
                filepath=filepath,
                import_shading='NORMALS',
                bone_heuristic='TEMPERANCE',
                guess_original_bind_pose=True
            )
            
            # Join all mesh objects if there are multiple
            mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
            if len(mesh_objects) > 1:
                # Select all mesh objects
                bpy.ops.object.select_all(action='DESELECT')
                for obj in mesh_objects:
                    obj.select_set(True)
                bpy.context.view_layer.objects.active = mesh_objects[0]
                # Join objects
                bpy.ops.object.join()
                
        elif file_ext == '.stl':
            bpy.ops.import_mesh.stl(filepath=filepath)
        elif file_ext == '.ply':
            bpy.ops.import_mesh.ply(filepath=filepath)
        elif file_ext == '.fbx':
            bpy.ops.import_scene.fbx(filepath=filepath)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}")
        
        # Get the imported object
        imported_obj = bpy.context.view_layer.objects.active
        
        # Validate imported object
        if not imported_obj:
            raise ValueError("No object was imported")
            
        if imported_obj.type != 'MESH':
            raise ValueError(f"Imported object is not a mesh (type: {imported_obj.type})")
            
        # Check if mesh has vertices
        if len(imported_obj.data.vertices) == 0:
            raise ValueError("Imported mesh has no vertices")
            
        # Print model info before normalization
        print_model_info(imported_obj, "BEFORE NORMALIZATION")
        
        # Normalize the model
        normalize_model(imported_obj)
        
        return imported_obj
        
    except Exception as e:
        print(f"Error during model import: {str(e)}")
        raise

def print_model_info(obj, stage=""):
    """Print model information for testing"""
    print(f"\n=== Model Info {stage} ===")
    print(f"Location: {obj.location}")
    print(f"Scale: {obj.scale}")
    print(f"Rotation: {obj.rotation_euler}")
    
    # Get bounding box dimensions
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    xs = [v[0] for v in bbox_corners]
    ys = [v[1] for v in bbox_corners]
    zs = [v[2] for v in bbox_corners]
    
    print(f"Bounding Box Dimensions:")
    print(f"X: {max(xs) - min(xs):.3f}")
    print(f"Y: {max(ys) - min(ys):.3f}")
    print(f"Z: {max(zs) - min(zs):.3f}")
    
    # Calculate distance from origin
    center = sum((mathutils.Vector(corner) for corner in bbox_corners), mathutils.Vector()) / 8
    print(f"Distance from origin: {center.length:.3f}") 