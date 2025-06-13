import bpy
import math
import os
import mathutils
import numpy as np
import json
from datetime import datetime

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


# Clear everything
bpy.ops.wm.read_factory_settings(use_empty=True)

# Set model name and paths
model_name = "becken"  # You can change this to any model name
model_path = f"C:/blender_test/3D_furniture_model/{model_name}.glb"
output_dir = f"C:/blender_test/2D_furniture_images/{model_name}"

# Create output directory structure
os.makedirs(output_dir, exist_ok=True)
views_dir = os.path.join(output_dir, "views").replace("\\", "/")
poses_dir = os.path.join(output_dir, "poses").replace("\\", "/")
os.makedirs(views_dir, exist_ok=True)
os.makedirs(poses_dir, exist_ok=True)

# Load 3D model
try:
    # Import and get the model
    imported_obj = import_3d_model(model_path)
    
    # Print model info after normalization
    print_model_info(imported_obj, "AFTER NORMALIZATION")
    
except Exception as e:
    print(f"Error importing model: {e}")
    raise

# Center and scale the mesh objects
imported_objects = [obj for obj in bpy.context.scene.objects if obj.type == 'MESH']
for obj in imported_objects:
    # Reset transformations
    obj.location = (0, 0, 0)
    obj.rotation_euler = (0, 0, 0)
    obj.scale = (1, 1, 1)
    
    # Apply all transformations
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

# Set render engine and resolution
bpy.context.scene.render.engine = 'CYCLES'
bpy.context.scene.cycles.device = 'GPU'

prefs = bpy.context.preferences
cycles_prefs = prefs.addons['cycles'].preferences
cycles_prefs.compute_device_type = 'OPTIX'  # Use 'CUDA' if OPTIX is not available

# Refresh device list
cycles_prefs.get_devices()

# Enable all GPU devices
for device in cycles_prefs.devices:
    device.use = True

bpy.context.scene.render.resolution_x = 1024
bpy.context.scene.render.resolution_y = 1024
bpy.context.scene.cycles.samples = 256  # Higher for better quality
bpy.context.scene.cycles.use_denoising = True

# Set color management for realism
bpy.context.scene.view_settings.view_transform = 'Filmic'
bpy.context.scene.view_settings.look = 'High Contrast'

# Set world background to light gray
if bpy.context.scene.world is None:
    bpy.context.scene.world = bpy.data.worlds.new("World")
bpy.context.scene.world.use_nodes = True
bg_tree = bpy.context.scene.world.node_tree
bg_nodes = bg_tree.nodes
bg_nodes.clear()
background = bg_nodes.new(type='ShaderNodeBackground')
background.inputs[0].default_value = (0.2, 0.2, 0.2, 1)  # Dark gray background
output = bg_nodes.new(type='ShaderNodeOutputWorld')
bg_tree.links.new(background.outputs[0], output.inputs[0])

# OPTIONAL: Add HDRI environment lighting (uncomment and set path if you have an HDRI)
# env_node = bg_nodes.new(type='ShaderNodeTexEnvironment')
# env_node.image = bpy.data.images.load('C:/blender_test/hdri/hdri.hdr')
# bg_tree.links.new(env_node.outputs['Color'], background.inputs['Color'])

# Add multiple lights for better illumination
bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
main_light = bpy.context.view_layer.objects.active
main_light.data.energy = 2.0  # Reduced from 8.0
main_light.data.angle = 0.1  # Sharper shadows

bpy.ops.object.light_add(type='AREA', location=(5, 5, 5))
fill_light = bpy.context.view_layer.objects.active
fill_light.data.energy = 1.0  # Reduced from 4.0
fill_light.data.size = 5.0  # Larger area light for softer fill

bpy.ops.object.light_add(type='AREA', location=(-5, -5, 5))
rim_light = bpy.context.view_layer.objects.active
rim_light.data.energy = 0.5  # Reduced from 2.0
rim_light.data.size = 3.0  # Smaller area light for rim effect

# Compute bounding box center and bounding sphere radius
bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for obj in imported_objects for corner in obj.bound_box]
center = [sum(coords) / len(coords) for coords in zip(*bbox_corners)]
radius = max((mathutils.Vector(center) - v).length for v in bbox_corners)

# Add camera and get the camera object
bpy.ops.object.camera_add()
cam = bpy.context.view_layer.objects.active
bpy.context.scene.camera = cam

# Set camera lens and sensor
cam.data.lens = 80  # mm
cam.data.sensor_width = 36  # mm
cam.data.clip_start = 0.01
cam.data.clip_end = 10000

# Get bounding box width and height in world coordinates
bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for obj in imported_objects for corner in obj.bound_box]
xs = [v[0] for v in bbox_corners]
ys = [v[1] for v in bbox_corners]
zs = [v[2] for v in bbox_corners]
bbox_width = max(xs) - min(xs)
bbox_depth = max(ys) - min(ys)
bbox_height = max(zs) - min(zs)

# Camera parameters
resolution_x = bpy.context.scene.render.resolution_x
resolution_y = bpy.context.scene.render.resolution_y
aspect_ratio = resolution_x / resolution_y
sensor_width = cam.data.sensor_width
sensor_height = sensor_width / aspect_ratio
focal_length = cam.data.lens

# Compute FOVs
fov_x = 2 * math.atan(sensor_width / (2 * focal_length))
fov_y = 2 * math.atan(sensor_height / (2 * focal_length))
fov = min(fov_x, fov_y)

# Use a large margin to guarantee the object is in view
cam_dist = radius / math.sin(fov / 2) * 0.9

num_elev = 5  # Number of elevation levels (vertical)
num_azim = 4  # Number of azimuth angles per elevation (horizontal)

elev_angles = np.linspace(-50, 80, num_elev)  # Avoid 0 and 90 to skip degenerate top/bottom
azim_angles = np.linspace(0, 360, num_azim, endpoint=False)

# Initialize views metadata
views_metadata = {
    "model_info": {
        "model_path": model_path.replace("\\", "/"),
        "model_name": model_name,
        "creation_date": datetime.now().isoformat(),
        "num_views": num_elev * num_azim
    },
    "camera_settings": {
        "resolution_x": bpy.context.scene.render.resolution_x,
        "resolution_y": bpy.context.scene.render.resolution_y,
        "focal_length": float(cam.data.lens),
        "sensor_width": float(cam.data.sensor_width)
    },
    "camera_intrinsics": {
        "fx": float(cam.data.lens * (bpy.context.scene.render.resolution_x / cam.data.sensor_width)),
        "fy": float(cam.data.lens * (bpy.context.scene.render.resolution_y / cam.data.sensor_width)),
        "cx": float(bpy.context.scene.render.resolution_x / 2),
        "cy": float(bpy.context.scene.render.resolution_y / 2)
    },
    "relative_poses": {
        "reference_view": 0,
        "transformations": []
    },
    "views": []
}

# Store all camera matrices for relative pose calculation
camera_matrices = []

view_idx = 0
for elev_deg in elev_angles:
    elev = math.radians(elev_deg)
    for azim_deg in azim_angles:
        azim = math.radians(azim_deg)
        try:
            cam_dir = mathutils.Vector((
                math.cos(azim) * math.cos(elev),
                math.sin(azim) * math.cos(elev),
                math.sin(elev)
            ))
            cam.location = mathutils.Vector(center) + cam_dir * cam_dist
            rot_quat = (mathutils.Vector(center) - cam.location).to_track_quat('-Z', 'Y')
            cam.rotation_euler = rot_quat.to_euler()
            
            # Update light positions relative to camera
            main_light.location = cam.location + mathutils.Vector((0, 0, 5))
            main_light.rotation_euler = cam.rotation_euler
            
            # Position fill light opposite to main light
            fill_offset = mathutils.Vector((3, -3, 2))
            fill_light.location = cam.location + fill_offset
            fill_light.rotation_euler = cam.rotation_euler
            
            # Position rim light for edge highlighting
            rim_offset = mathutils.Vector((-2, 2, 1))
            rim_light.location = cam.location + rim_offset
            rim_light.rotation_euler = cam.rotation_euler
            
            # Generate view paths with new directory structure
            view_image_path = os.path.join(views_dir, f"view_{view_idx:03}.png").replace("\\", "/")
            view_pose_path = os.path.join(poses_dir, f"pose_{view_idx:03}.txt").replace("\\", "/")
            
            # Render the view
            bpy.context.scene.render.filepath = view_image_path
            try:
                bpy.ops.render.render(write_still=True)
            except Exception as e:
                print(f"Render failed at view {view_idx}: {str(e)}")
                continue
            
            # Save pose matrix
            try:
                matrix = cam.matrix_world
                camera_matrices.append(matrix)  # Store matrix for relative pose calculation
                with open(view_pose_path, 'w') as f:
                    for row in matrix:
                        f.write(" ".join([str(v) for v in row]) + "\n")
            except Exception as e:
                print(f"Failed to save pose file for view {view_idx}: {str(e)}")
                continue
            
            # Add view metadata with relative paths
            view_metadata = {
                "view_index": view_idx,
                "image_path": os.path.relpath(view_image_path, output_dir).replace("\\", "/"),
                "pose_path": os.path.relpath(view_pose_path, output_dir).replace("\\", "/"),
                "elevation_degrees": float(elev_deg),
                "azimuth_degrees": float(azim_deg)
            }
            views_metadata["views"].append(view_metadata)
            
            view_idx += 1
            
        except Exception as e:
            print(f"Error processing view {view_idx}: {str(e)}")
            continue

# Calculate relative poses between all views
reference_view = 0
for i in range(len(camera_matrices)):
    for j in range(len(camera_matrices)):
        if i != j:
            # Calculate relative transform
            relative_transform = np.linalg.inv(camera_matrices[i]) @ camera_matrices[j]
            # Convert to list for JSON serialization
            relative_transform_list = relative_transform.tolist()
            
            # Add to transformations
            views_metadata["relative_poses"]["transformations"].append({
                "from_view": i,
                "to_view": j,
                "transform_matrix": relative_transform_list
            })

# Save views metadata to JSON
try:
    views_json_path = os.path.join(output_dir, "views.json").replace("\\", "/")
    with open(views_json_path, 'w') as f:
        json.dump(views_metadata, f, indent=2)
    print(f"\nMetadata saved to: {views_json_path}")
except Exception as e:
    print(f"Failed to save views.json: {str(e)}")

# Create dataset splits
try:
    # Create dataset splits
    all_views = views_metadata["views"]
    n_views = len(all_views)
    
    # Calculate split sizes (70% train, 15% val, 15% test)
    train_size = int(n_views * 0.7)
    val_size = int(n_views * 0.15)
    
    # Shuffle views
    np.random.seed(42)  # For reproducibility
    indices = np.random.permutation(n_views)
    
    # Create splits
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    # Create split data
    dataset_splits = {
        "train": [
            {
                "image": all_views[i]["image_path"].replace("\\", "/"),
                "pose": all_views[i]["pose_path"].replace("\\", "/")
            }
            for i in train_indices
        ],
        "val": [
            {
                "image": all_views[i]["image_path"].replace("\\", "/"),
                "pose": all_views[i]["pose_path"].replace("\\", "/")
            }
            for i in val_indices
        ],
        "test": [
            {
                "image": all_views[i]["image_path"].replace("\\", "/"),
                "pose": all_views[i]["pose_path"].replace("\\", "/")
            }
            for i in test_indices
        ]
    }
    
    # Save dataset splits
    splits_json_path = os.path.join(output_dir, "dataset_splits.json").replace("\\", "/")
    with open(splits_json_path, 'w') as f:
        json.dump(dataset_splits, f, indent=2)
    print(f"\nDataset splits saved to: {splits_json_path}")
    
    # Print split statistics
    print("\nDataset Split Statistics:")
    print(f"Total views: {n_views}")
    print(f"Train set: {len(dataset_splits['train'])} views")
    print(f"Validation set: {len(dataset_splits['val'])} views")
    print(f"Test set: {len(dataset_splits['test'])} views")
    
except Exception as e:
    print(f"Failed to create dataset splits: {str(e)}")
