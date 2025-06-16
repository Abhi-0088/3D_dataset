import bpy
import os
import mathutils
import numpy as np

def setup_normal_render():
    """Set up render settings and node tree for normal map generation"""
    scene = bpy.context.scene
    
    print("\n=== Setting up normal render ===")
    
    # Set render settings
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.cycles.samples = 256
    scene.cycles.use_denoising = True
    
    # Set render output format to EXR
    scene.render.image_settings.file_format = 'OPEN_EXR'
    scene.render.image_settings.color_mode = 'RGB'
    scene.render.image_settings.color_depth = '32'
    scene.render.image_settings.use_zbuffer = True
    
    print("Render settings configured")
    
    # Enable normal pass in view layer
    view_layer = scene.view_layers[0]
    view_layer.use_pass_normal = True
    print("Normal pass enabled in view layer")
    
    # Set up node tree
    scene.use_nodes = True
    tree = scene.node_tree
    links = tree.links
    
    # Clear existing nodes
    tree.nodes.clear()
    print("Cleared existing nodes")
    
    # Create render layers node
    render_layers = tree.nodes.new('CompositorNodeRLayers')
    render_layers.location = (-300, 0)
    print("Created render layers node")
    
    # Create output node
    output = tree.nodes.new('CompositorNodeOutputFile')
    output.location = (300, 0)
    output.format.file_format = 'OPEN_EXR'
    output.format.color_mode = 'RGB'
    output.format.color_depth = '32'
    print("Created output node")
    
    # Find the normal output socket
    normal_output = None
    for i, output_socket in enumerate(render_layers.outputs):
        if output_socket.name == 'Normal':
            normal_output = output_socket
            print(f"\nFound normal output at index {i}")
            break
    
    if normal_output:
        try:
            # Create a new link between the normal output and the file output
            new_link = links.new(normal_output, output.inputs[0])
            print(f"Successfully created link from {normal_output.name} to output")
        except Exception as e:
            print(f"Error creating link: {str(e)}")
            print(f"Normal output type: {type(normal_output)}")
            print(f"Output input type: {type(output.inputs[0])}")
            raise
    else:
        raise Exception("No normal output found in render layers node")
    
    return output

def generate_normal_map(obj, output_path):
    """Generate a normal map for the given object"""
    print(f"\n=== Generating normal map for {obj.name} ===")
    print(f"Output path: {output_path}")
    
    # Save original scene state
    print("Saving original scene state...")
    original_state = save_scene_state()
    
    # Store original materials
    original_materials = {}
    for slot in obj.material_slots:
        if slot.material:
            original_materials[slot.name] = slot.material
    print(f"Stored {len(original_materials)} original materials")
    
    try:
        # Set up normal render
        print("Setting up normal render...")
        normal_output = setup_normal_render()
        
        # Set output path
        print(f"Setting output path: {output_path}")
        output_dir = os.path.dirname(output_path)
        output_name = os.path.splitext(os.path.basename(output_path))[0]
        normal_output.base_path = output_dir
        normal_output.file_slots[0].path = output_name
        normal_output.file_slots[0].use_node_format = True
        normal_output.file_slots[0].format.file_format = 'OPEN_EXR'
        normal_output.file_slots[0].format.color_mode = 'RGB'
        normal_output.file_slots[0].format.color_depth = '32'
        
        # Create normal map material
        print("Creating normal map material...")
        normal_mat = bpy.data.materials.new(name="NormalMapMaterial")
        normal_mat.use_nodes = True
        nodes = normal_mat.node_tree.nodes
        links = normal_mat.node_tree.links
        
        # Clear default nodes
        nodes.clear()
        
        # Create geometry node for world space normals
        geometry = nodes.new('ShaderNodeNewGeometry')
        geometry.location = (-300, 0)
        
        # Create vector transform node to convert to world space
        vector_transform = nodes.new('ShaderNodeVectorTransform')
        vector_transform.location = (-100, 0)
        vector_transform.convert_from = 'CAMERA'
        vector_transform.convert_to = 'WORLD'
        
        # Create output node
        output = nodes.new('ShaderNodeOutputMaterial')
        output.location = (100, 0)
        
        # Link nodes
        links.new(geometry.outputs['Normal'], vector_transform.inputs[0])
        links.new(vector_transform.outputs[0], output.inputs[0])
        
        # Apply normal map material
        obj.data.materials.clear()
        obj.data.materials.append(normal_mat)
        print("Applied normal map material")
        
        # Store original render filepath
        original_render_path = bpy.context.scene.render.filepath
        
        # Disable render output to prevent double saving
        bpy.context.scene.render.filepath = ""
        bpy.context.scene.render.image_settings.file_format = 'OPEN_EXR'
        bpy.context.scene.render.image_settings.color_mode = 'RGB'
        bpy.context.scene.render.image_settings.color_depth = '32'
        
        # Render normal map
        print(f"Rendering normal map to: {output_path}")
        bpy.ops.render.render(write_still=True)
        print("Render completed")
        
        # Restore original render filepath
        bpy.context.scene.render.filepath = original_render_path
        
        # Get the actual saved normal map path
        actual_normal_path = os.path.join(output_dir, f"{output_name}0001.exr").replace("\\", "/")
        if os.path.exists(actual_normal_path):
            # Remove existing file if it exists
            if os.path.exists(output_path):
                os.remove(output_path)
            # Rename the file to the desired name
            os.replace(actual_normal_path, output_path)
            print(f"Renamed normal map from {actual_normal_path} to {output_path}")
        else:
            print(f"Warning: Could not find normal map at {actual_normal_path}")
            return None
        
        # Restore original materials first
        obj.data.materials.clear()
        for slot_name, material in original_materials.items():
            obj.data.materials.append(material)
        print("Restored original materials")
        
        # Clean up temporary material
        if 'NormalMapMaterial' in bpy.data.materials:
            bpy.data.materials.remove(bpy.data.materials['NormalMapMaterial'])
        print("Removed temporary material")
        
        # Restore original scene state
        restore_scene_state(original_state)
        print("Restored original scene state")
        
        print("Cleanup completed")
        return output_path
        
    except Exception as e:
        print(f"Error generating normal map: {str(e)}")
        import traceback
        print("Full error traceback:")
        print(traceback.format_exc())
        return None
        
    finally:
        # Ensure cleanup happens even if there was an error
        try:
            # Restore original materials
            obj.data.materials.clear()
            for slot_name, material in original_materials.items():
                obj.data.materials.append(material)
            print("Restored original materials")
            
            # Clean up temporary material
            if 'NormalMapMaterial' in bpy.data.materials:
                bpy.data.materials.remove(bpy.data.materials['NormalMapMaterial'])
            print("Removed temporary material")
            
            # Restore original scene state
            restore_scene_state(original_state)
            print("Restored original scene state")
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
        
        print("Cleanup completed")

def save_scene_state():
    """Save the current scene state"""
    scene = bpy.context.scene
    
    # Save render settings
    render_settings = {
        'engine': scene.render.engine,
        'samples': scene.cycles.samples,
        'use_denoising': scene.cycles.use_denoising,
        'device': scene.cycles.device,
        'resolution_x': scene.render.resolution_x,
        'resolution_y': scene.render.resolution_y
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

def restore_scene_state(state):
    """Restore the saved scene state"""
    scene = bpy.context.scene
    
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
                    
                    # Copy output values
                    for i, output in enumerate(node.outputs):
                        if i < len(new_node.outputs):
                            try:
                                new_node.outputs[i].default_value = output.default_value
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

def process_normal_map(normal_img):
    """Process normal map to ensure it's in world space"""
    # Get normal pixels and reshape
    normal_pixels = np.array(normal_img.pixels)
    normal_map = normal_pixels.reshape(normal_img.size[1], normal_img.size[0], 4)[:, :, :3]  # Take RGB channels
    
    # Normalize the normals to ensure they are unit vectors
    normal_lengths = np.linalg.norm(normal_map, axis=2, keepdims=True)
    normal_map = np.divide(normal_map, normal_lengths, where=normal_lengths!=0)
    
    # Create new image for processed normals
    processed_normal_img = bpy.data.images.new(
        name=f"world_space_normal",
        width=normal_img.size[0],
        height=normal_img.size[1],
        alpha=False,
        float_buffer=True
    )
    
    # Convert to RGBA format (add alpha channel)
    normal_rgba = np.concatenate([normal_map, np.ones((*normal_map.shape[:2], 1))], axis=2)
    processed_normal_img.pixels = normal_rgba.ravel()
    
    return processed_normal_img 