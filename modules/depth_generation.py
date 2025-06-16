import bpy
import numpy as np
import mathutils

def setup_depth_render():
    """Set up render settings for depth map generation"""
    # Set render engine and device
    bpy.context.scene.render.engine = 'CYCLES'
    bpy.context.scene.cycles.device = 'GPU'
    
    # Enable depth pass
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
    
    # Set world background to dark gray
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

def process_depth_map(depth_img, K, R, t):
    """Process depth map to get true Euclidean depth"""
    # Get depth pixels and reshape
    depth_pixels = np.array(depth_img.pixels)
    depth_map = depth_pixels.reshape(depth_img.size[1], depth_img.size[0], 4)[:, :, 0]  # Take only R channel
    
    # Print raw depth map statistics
    print(f"Raw depth map stats - Min: {np.min(depth_map)}, Max: {np.max(depth_map)}, Mean: {np.mean(depth_map)}")
    
    # Convert to true Euclidean depth
    true_depth = zbuffer_to_euclidean_depth(depth_map, K, R, t)
    
    # Create new image for true Euclidean depth
    true_depth_img = bpy.data.images.new(
        name=f"euclidean_depth",
        width=depth_img.size[0],
        height=depth_img.size[1],
        alpha=False,
        float_buffer=True
    )
    
    # Convert to RGBA format (repeat the depth value for all channels)
    true_depth_rgba = np.stack([true_depth] * 4, axis=-1)
    true_depth_img.pixels = true_depth_rgba.ravel()
    
    # Print final depth map statistics
    print(f"Final depth map stats - Min: {np.min(true_depth)}, Max: {np.max(true_depth)}, Mean: {np.mean(true_depth)}")
    
    return true_depth_img

def zbuffer_to_euclidean_depth(depth_map, K, R, t):
    """
    Convert Z-buffer depth to true Euclidean depth
    Args:
        depth_map: Z-buffer depth map
        K: Camera intrinsics matrix
        R: Camera rotation matrix
        t: Camera translation vector
    Returns:
        true_depth: True Euclidean depth map
    """
    h, w = depth_map.shape
    i, j = np.meshgrid(np.arange(w), np.arange(h))
    
    # Convert pixel coords to normalized camera coordinates
    x = (i - K[0, 2]) / K[0, 0]
    y = (j - K[1, 2]) / K[1, 1]
    
    # Z-buffer distance
    z = depth_map
    
    # Print depth statistics for debugging
    print(f"Depth map stats - Min: {np.min(z)}, Max: {np.max(z)}, Mean: {np.mean(z)}")
    
    # Camera space points
    Xc = np.stack([x * z, y * z, z], axis=-1)
    
    # Convert to world coordinates
    Xw = (R.T @ Xc.reshape(-1, 3).T - (R.T @ t.reshape(-1, 1))).T
    
    # Euclidean depth from camera origin
    true_depth = np.linalg.norm(Xw, axis=1).reshape(h, w)
    
    # Print true depth statistics for debugging
    print(f"True depth stats - Min: {np.min(true_depth)}, Max: {np.max(true_depth)}, Mean: {np.mean(true_depth)}")
    
    # Normalize the depth values to [0, 1] range for visualization
    depth_min = np.min(true_depth)
    depth_max = np.max(true_depth)
    if depth_max > depth_min:
        true_depth = (true_depth - depth_min) / (depth_max - depth_min)
    
    return true_depth 