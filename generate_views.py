import bpy
import math
import os
import mathutils
import numpy as np
import json
from datetime import datetime
import sys
import time

# Add the current directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.append(current_dir)

# Use absolute imports instead of relative imports
from modules.model_import import import_3d_model, normalize_model, print_model_info
from modules.view_generation import setup_camera, setup_lights, generate_camera_views
from modules.depth_generation import setup_depth_render, process_depth_map, zbuffer_to_euclidean_depth
from modules.pose_estimation import calculate_camera_intrinsics, calculate_relative_poses, save_pose_data, get_camera_pose
from modules.utils import ensure_directory, save_metadata, generate_metadata, print_matrix
from modules.camera_setup import create_camera_and_lights
from modules.normal_generation import generate_normal_map

def save_view_state():
    """Save the current view state including render settings and node tree"""
    scene = bpy.context.scene
    
    # Save render settings
    render_settings = {
        'engine': scene.render.engine,
        'samples': scene.cycles.samples,
        'use_denoising': scene.cycles.use_denoising,
        'device': scene.cycles.device,
        'resolution_x': scene.render.resolution_x,
        'resolution_y': scene.render.resolution_y,
        'filepath': scene.render.filepath  # Save the current filepath
    }
    
    # Save node tree state
    node_tree = None
    if scene.use_nodes:
        node_tree = scene.node_tree.copy()
    
    # Save view layer settings
    view_layer = scene.view_layers[0]
    view_layer_settings = {
        'use_pass_z': view_layer.use_pass_z,
        'use_pass_mist': view_layer.use_pass_mist,
        'use_pass_normal': view_layer.use_pass_normal
    }
    
    return {
        'render_settings': render_settings,
        'node_tree': node_tree,
        'view_layer_settings': view_layer_settings
    }

def restore_view_state(state):
    """Restore the saved view state"""
    scene = bpy.context.scene
    
    # Store current filepath
    current_filepath = scene.render.filepath
    
    # Restore render settings
    scene.render.engine = state['render_settings']['engine']
    scene.cycles.samples = state['render_settings']['samples']
    scene.cycles.use_denoising = state['render_settings']['use_denoising']
    scene.cycles.device = state['render_settings']['device']
    scene.render.resolution_x = state['render_settings']['resolution_x']
    scene.render.resolution_y = state['render_settings']['resolution_y']
    
    # Restore node tree
    if scene.use_nodes:
        scene.node_tree.nodes.clear()
        scene.node_tree.links.clear()
        if state['node_tree']:
            # Map of old node names to new nodes
            node_map = {}
            
            # Node type mapping
            node_type_map = {
                'R_LAYERS': 'CompositorNodeRLayers',
                'OUTPUT_FILE': 'CompositorNodeOutputFile',
                'NORMALIZE': 'CompositorNodeNormalize',
                'NORMAL_MAP': 'ShaderNodeNormalMap',
                'OUTPUT_MATERIAL': 'ShaderNodeOutputMaterial'
            }
            
            # First pass: create all nodes
            for node in state['node_tree'].nodes:
                try:
                    # Get the correct node type
                    node_type = node_type_map.get(node.type, node.type)
                    new_node = scene.node_tree.nodes.new(type=node_type)
                    new_node.location = node.location
                    new_node.width = node.width
                    new_node.height = node.height
                    
                    # Store mapping from old node to new node
                    node_map[node.name] = new_node
                    
                    # Copy input values
                    for i, input in enumerate(node.inputs):
                        if i < len(new_node.inputs):
                            try:
                                new_node.inputs[i].default_value = input.default_value
                            except:
                                pass
                except Exception as e:
                    print(f"Warning: Could not restore node {node.name}: {str(e)}")
                    continue
            
            # Second pass: restore links
            for link in state['node_tree'].links:
                try:
                    from_node = node_map.get(link.from_node.name)
                    to_node = node_map.get(link.to_node.name)
                    
                    if from_node and to_node:
                        # Find matching sockets
                        from_socket = None
                        to_socket = None
                        
                        # Try to find matching output socket
                        for output in from_node.outputs:
                            if output.name == link.from_socket.name:
                                from_socket = output
                                break
                        
                        # Try to find matching input socket
                        for input in to_node.inputs:
                            if input.name == link.to_socket.name:
                                to_socket = input
                                break
                        
                        if from_socket and to_socket:
                            scene.node_tree.links.new(from_socket, to_socket)
                except Exception as e:
                    print(f"Warning: Could not restore link: {str(e)}")
                    continue
    
    # Restore view layer settings
    view_layer = scene.view_layers[0]
    view_layer.use_pass_z = state['view_layer_settings']['use_pass_z']
    view_layer.use_pass_mist = state['view_layer_settings']['use_pass_mist']
    view_layer.use_pass_normal = state['view_layer_settings']['use_pass_normal']
    
    # Restore the original filepath
    scene.render.filepath = current_filepath

def main():
    # Clear existing scene
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete()
    
    # Set model name and paths
    model_name = "becken"  # You can change this to any model name
    model_path = f"C:/blender_test/3D_furniture_model/{model_name}.glb"
    output_dir = f"C:/blender_test/2D_furniture_images/{model_name}"
    
    # Create output directory structure
    os.makedirs(output_dir, exist_ok=True)
    views_dir = os.path.join(output_dir, "views").replace("\\", "/")
    poses_dir = os.path.join(output_dir, "poses").replace("\\", "/")
    depth_maps_dir = os.path.join(output_dir, "depth_maps").replace("\\", "/")
    normal_maps_dir = os.path.join(output_dir, "normal_maps").replace("\\", "/")
    os.makedirs(views_dir, exist_ok=True)
    os.makedirs(poses_dir, exist_ok=True)
    os.makedirs(depth_maps_dir, exist_ok=True)
    os.makedirs(normal_maps_dir, exist_ok=True)
    
    # Import and normalize model
    obj = import_3d_model(model_path)
    if obj is None:
        print("Failed to import model")
        return
    
    normalize_model(obj)
    print_model_info(obj, "After normalization")
    
    # Compute bounding box center and bounding sphere radius
    bbox_corners = [obj.matrix_world @ mathutils.Vector(corner) for corner in obj.bound_box]
    center = [sum(coords) / len(coords) for coords in zip(*bbox_corners)]
    radius = max((mathutils.Vector(center) - v).length for v in bbox_corners)
    
    # Set render engine and resolution
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    
    # Enable depth pass for depth map generation
    bpy.context.scene.view_layers[0].use_pass_z = True
    bpy.context.scene.view_layers[0].use_pass_mist = True
    
    # Set up render layers for depth
    bpy.context.scene.use_nodes = True
    tree = bpy.context.scene.node_tree
    links = tree.links
    
    # Clear default nodes
    for node in tree.nodes:
        tree.nodes.remove(node)
    
    # Create input render layer node
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    
    # Create normalize node to scale depth values
    normalize = tree.nodes.new('CompositorNodeNormalize')
    normalize.inputs[0].default_value = 1.0
    
    # Create output node
    depth_output = tree.nodes.new('CompositorNodeOutputFile')
    depth_output.format.file_format = 'OPEN_EXR'
    depth_output.format.color_mode = 'RGB'
    depth_output.format.color_depth = '32'
    depth_output.format.use_zbuffer = True
    
    # Link nodes
    links.new(render_layers.outputs['Depth'], normalize.inputs[0])
    links.new(normalize.outputs[0], depth_output.inputs[0])
    
    # Set up GPU compute device
    prefs = bpy.context.preferences
    cycles_prefs = prefs.addons['cycles'].preferences
    cycles_prefs.compute_device_type = 'OPTIX'  # Use 'CUDA' if OPTIX is not available
    
    # Refresh device list
    cycles_prefs.get_devices()
    
    # Enable all GPU devices
    for device in cycles_prefs.devices:
        device.use = True
    
    # Set render settings
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
    
    # Set up initial camera for metadata
    bpy.ops.object.camera_add(location=(0, 0, 0))
    initial_camera = bpy.context.active_object
    initial_camera.name = "InitialCamera"
    bpy.context.scene.camera = initial_camera
    
    # Set camera parameters
    initial_camera.data.lens = 80  # mm
    initial_camera.data.sensor_width = 36  # mm
    initial_camera.data.clip_start = 0.1
    initial_camera.data.clip_end = 100.0
    
    # Generate camera views
    center = np.array(center)  # Use computed center
    num_elev = 3  # Match original
    num_azim = 2  # Match original
    views = generate_camera_views(center, radius, num_elev, num_azim)
    
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
            "focal_length": float(initial_camera.data.lens),
            "sensor_width": float(initial_camera.data.sensor_width)
        },
        "camera_intrinsics": {
            "fx": float(initial_camera.data.lens * (bpy.context.scene.render.resolution_x / initial_camera.data.sensor_width)),
            "fy": float(initial_camera.data.lens * (bpy.context.scene.render.resolution_y / initial_camera.data.sensor_width)),
            "cx": float(bpy.context.scene.render.resolution_x / 2),
            "cy": float(bpy.context.scene.render.resolution_y / 2)
        },
        "relative_poses": {
            "reference_view": 0,
            "transformations": []
        },
        "views": []
    }
    
    # Remove initial camera as we'll create new ones for each view
    bpy.data.objects.remove(initial_camera)
    
    # Save initial scene state
    initial_scene_state = save_view_state()
    
    # Process each view
    camera_matrices = []
    for i, view in enumerate(views):
        print(f"\nProcessing view {i}")
        
        # Save the initial state before any view-specific setup
        view_state = save_view_state()
        
        try:
            # Setup camera for this view
            cam, cam_distance = setup_camera(center, radius, view["fov"])
            cam.location = view["location"]
            cam.rotation_euler = view["rotation"]
            
            # Setup lights for this view
            main_light, fill_light, rim_light = setup_lights(view["location"], view["rotation"])
            
            # Get camera pose
            R, t = get_camera_pose(cam)
            camera_matrix = np.eye(4)
            camera_matrix[:3, :3] = R
            camera_matrix[:3, 3] = t
            camera_matrices.append(camera_matrix)
            
            # Save pose immediately after getting it
            view_pose_path = os.path.join(poses_dir, f"pose_{i:03d}.txt").replace("\\", "/")
            with open(view_pose_path, 'w') as f:
                for row in camera_matrix:
                    f.write(" ".join([str(v) for v in row]) + "\n")
            
            # Calculate intrinsics
            K = calculate_camera_intrinsics(cam, bpy.context.scene.render.resolution_x, 
                                          bpy.context.scene.render.resolution_y)
            
            # Generate view paths
            view_image_path = os.path.join(views_dir, f"view_{i:03d}.png").replace("\\", "/")
            depth_map_path = os.path.join(depth_maps_dir, f"depth_{i:03d}.exr").replace("\\", "/")
            euclidean_depth_path = os.path.join(depth_maps_dir, f"euclidean_depth_{i:03d}.exr").replace("\\", "/")
            normal_map_path = os.path.join(normal_maps_dir, f"normal_{i:03d}.exr").replace("\\", "/")
            
            # First render for RGB image
            print(f"Rendering RGB image: {view_image_path}")
            
            # Store original render settings
            original_render_path = bpy.context.scene.render.filepath
            original_format = bpy.context.scene.render.image_settings.file_format
            
            # Set output path for RGB render
            bpy.context.scene.render.filepath = view_image_path
            bpy.context.scene.render.image_settings.file_format = 'PNG'
            
            # Disable compositing for RGB render
            bpy.context.scene.use_nodes = False
            
            # Render RGB image
            bpy.ops.render.render(write_still=True)
            
            # Restore original render settings
            bpy.context.scene.render.filepath = original_render_path
            bpy.context.scene.render.image_settings.file_format = original_format
            
            # Ensure the file is completely written before proceeding
            time.sleep(0.5)  # Give the system time to finish writing the file
            
            # Set up depth output path and render
            print(f"Rendering depth map: {depth_map_path}")
            
            # Enable compositing for depth render
            bpy.context.scene.use_nodes = True
            
            # Clear any existing nodes
            bpy.context.scene.node_tree.nodes.clear()
            bpy.context.scene.node_tree.links.clear()
            
            # Create render layers node
            render_layers = bpy.context.scene.node_tree.nodes.new('CompositorNodeRLayers')
            render_layers.location = (-300, 0)
            
            # Create normalize node
            normalize = bpy.context.scene.node_tree.nodes.new('CompositorNodeNormalize')
            normalize.location = (0, 0)
            
            # Create output node
            depth_output = bpy.context.scene.node_tree.nodes.new('CompositorNodeOutputFile')
            depth_output.location = (300, 0)
            depth_output.format.file_format = 'OPEN_EXR'
            depth_output.format.color_mode = 'RGB'
            depth_output.format.color_depth = '32'
            depth_output.format.use_zbuffer = True
            
            # Link nodes
            bpy.context.scene.node_tree.links.new(render_layers.outputs['Depth'], normalize.inputs[0])
            bpy.context.scene.node_tree.links.new(normalize.outputs[0], depth_output.inputs[0])
            
            # Set the base path and file name
            depth_output.base_path = depth_maps_dir
            depth_output.file_slots[0].path = f"depth_{i:03d}"
            depth_output.file_slots[0].use_node_format = True
            depth_output.file_slots[0].format.file_format = 'OPEN_EXR'
            depth_output.file_slots[0].format.color_mode = 'RGB'
            depth_output.file_slots[0].format.color_depth = '32'
            depth_output.file_slots[0].format.use_zbuffer = True
            
            # Store original render settings
            original_render_path = bpy.context.scene.render.filepath
            original_format = bpy.context.scene.render.image_settings.file_format
            
            # Disable render output completely
            bpy.context.scene.render.filepath = ""
            bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
            bpy.context.scene.render.image_settings.color_mode = 'RGB'
            bpy.context.scene.render.image_settings.color_depth = '32'
            bpy.context.scene.render.image_settings.use_zbuffer = True
            
            # Render depth map
            bpy.ops.render.render(write_still=True)
            
            # Restore original render settings
            bpy.context.scene.render.filepath = original_render_path
            bpy.context.scene.render.image_settings.file_format = original_format
            
            # Ensure the file is completely written before proceeding
            time.sleep(0.5)
            
            # Get the actual saved depth map path
            actual_depth_path = os.path.join(depth_maps_dir, f"depth_{i:03d}0001.exr").replace("\\", "/")
            if os.path.exists(actual_depth_path):
                # Remove existing file if it exists
                if os.path.exists(depth_map_path):
                    os.remove(depth_map_path)
                # Rename the file to the desired name
                os.replace(actual_depth_path, depth_map_path)
                print(f"Renamed depth map from {actual_depth_path} to {depth_map_path}")
            else:
                print(f"Warning: Could not find depth map at {actual_depth_path}")
                continue
            
            # Read depth map using Blender's image handling
            depth_img = bpy.data.images.load(depth_map_path)
            
            # Process depth map
            processed_depth = process_depth_map(depth_img, K, R, t)
            
            # Save processed depth map
            print(f"Saving Euclidean depth map: {euclidean_depth_path}")
            processed_depth.save_render(euclidean_depth_path)
            
            # Verify the file was saved
            if os.path.exists(euclidean_depth_path):
                print(f"Successfully saved Euclidean depth map to: {euclidean_depth_path}")
            else:
                print(f"Warning: Failed to save Euclidean depth map to: {euclidean_depth_path}")
            
            # Clean up
            bpy.data.images.remove(depth_img)
            bpy.data.images.remove(processed_depth)
            
            # Ensure the file is completely written before proceeding
            time.sleep(0.5)  # Give the system time to finish writing the file
            
            # Generate normal map
            try:
                # Save current render filepath and node tree state
                current_render_path = bpy.context.scene.render.filepath
                current_node_tree = bpy.context.scene.node_tree.copy()
                
                # Generate normal map
                print(f"Generating normal map for view {i}")
                normal_map_path = generate_normal_map(obj, normal_map_path)
                
                # Restore render filepath and node tree
                bpy.context.scene.render.filepath = current_render_path
                restore_view_state(view_state)  # Use the saved view state instead of manual restoration
                
                if normal_map_path is None:
                    print(f"Warning: Failed to generate normal map for view {i}")
                    normal_map_path = ""  # Set to empty string if generation failed
            except Exception as e:
                print(f"Warning: Error generating normal map for view {i}: {str(e)}")
                normal_map_path = ""  # Set to empty string if generation failed
                
        except Exception as e:
            print(f"Render failed at view {i}: {str(e)}")
            continue
            
        finally:
            # Clean up objects for this view
            if 'cam' in locals():
                bpy.data.objects.remove(cam)
            if 'main_light' in locals():
                bpy.data.objects.remove(main_light)
            if 'fill_light' in locals():
                bpy.data.objects.remove(fill_light)
            if 'rim_light' in locals():
                bpy.data.objects.remove(rim_light)
            
            # Restore the view state
            restore_view_state(view_state)
            
            # Ensure we're back to the initial scene state
            restore_view_state(initial_scene_state)
            
            # Ensure the file is completely written before proceeding
            time.sleep(0.5)  # Give the system time to finish writing the file
            
            # Add view metadata
            view_metadata = {
                "view_index": i,
                "image_path": os.path.relpath(view_image_path, output_dir).replace("\\", "/"),
                "pose_path": os.path.relpath(view_pose_path, output_dir).replace("\\", "/"),
                "depth_map_path": os.path.relpath(depth_map_path, output_dir).replace("\\", "/"),
                "euclidean_depth_path": os.path.relpath(euclidean_depth_path, output_dir).replace("\\", "/"),
                "normal_map_path": os.path.relpath(normal_map_path, output_dir).replace("\\", "/") if normal_map_path else "",
                "elevation_degrees": float(view["elevation"]),
                "azimuth_degrees": float(view["azimuth"])
            }
            views_metadata["views"].append(view_metadata)
    
    # Calculate relative poses
    reference_view = 0
    for i in range(len(camera_matrices)):
        for j in range(len(camera_matrices)):
            if i != j:
                relative_transform = np.linalg.inv(camera_matrices[i]) @ camera_matrices[j]
                views_metadata["relative_poses"]["transformations"].append({
                    "from_view": i,
                    "to_view": j,
                    "transform_matrix": relative_transform.tolist()
                })
    
    # Save views metadata
    views_json_path = os.path.join(output_dir, "views.json").replace("\\", "/")
    with open(views_json_path, 'w') as f:
        json.dump(views_metadata, f, indent=2)
    print(f"\nMetadata saved to: {views_json_path}")
    
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
                "pose": all_views[i]["pose_path"].replace("\\", "/"),
                "depth_map": all_views[i]["depth_map_path"].replace("\\", "/"),
                "euclidean_depth": all_views[i]["euclidean_depth_path"].replace("\\", "/"),
                "normal_map": all_views[i]["normal_map_path"].replace("\\", "/")
            }
            for i in train_indices
        ],
        "val": [
            {
                "image": all_views[i]["image_path"].replace("\\", "/"),
                "pose": all_views[i]["pose_path"].replace("\\", "/"),
                "depth_map": all_views[i]["depth_map_path"].replace("\\", "/"),
                "euclidean_depth": all_views[i]["euclidean_depth_path"].replace("\\", "/"),
                "normal_map": all_views[i]["normal_map_path"].replace("\\", "/")
            }
            for i in val_indices
        ],
        "test": [
            {
                "image": all_views[i]["image_path"].replace("\\", "/"),
                "pose": all_views[i]["pose_path"].replace("\\", "/"),
                "depth_map": all_views[i]["depth_map_path"].replace("\\", "/"),
                "euclidean_depth": all_views[i]["euclidean_depth_path"].replace("\\", "/"),
                "normal_map": all_views[i]["normal_map_path"].replace("\\", "/")
            }
            for i in test_indices
        ]
    }
    
    # Save dataset splits
    splits_json_path = os.path.join(output_dir, "dataset_splits.json").replace("\\", "/")
    with open(splits_json_path, 'w') as f:
        json.dump(dataset_splits, f, indent=2)
    print(f"\nDataset splits saved to: {splits_json_path}")
    
    print("\nDataset Split Statistics:")
    print(f"Total views: {n_views}")
    print(f"Train set: {len(dataset_splits['train'])} views")
    print(f"Validation set: {len(dataset_splits['val'])} views")
    print(f"Test set: {len(dataset_splits['test'])} views")

if __name__ == "__main__":
    main()
