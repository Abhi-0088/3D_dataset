import bpy
import math
import mathutils
import numpy as np

def setup_camera(center, radius, fov):
    """Set up camera with correct parameters"""
    # Add camera
    bpy.ops.object.camera_add()
    cam = bpy.context.view_layer.objects.active
    bpy.context.scene.camera = cam
    
    # Set camera parameters
    cam.data.lens = 80  # mm
    cam.data.sensor_width = 36  # mm
    cam.data.clip_start = 0.1
    cam.data.clip_end = 100.0
    
    # Calculate camera distance
    cam_dist = radius / math.sin(fov / 2) * 0.9  # Match original code exactly
    
    return cam, cam_dist

def setup_lights(camera_location, camera_rotation):
    """Set up lighting for the scene"""
    # Main light (sun)
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 10))
    main_light = bpy.context.view_layer.objects.active
    main_light.data.energy = 2.0  # Reduced from 8.0
    main_light.data.angle = 0.1  # Sharper shadows
    
    # Fill light (area)
    bpy.ops.object.light_add(type='AREA', location=(5, 5, 5))
    fill_light = bpy.context.view_layer.objects.active
    fill_light.data.energy = 1.0  # Reduced from 4.0
    fill_light.data.size = 5.0  # Larger area light for softer fill
    
    # Rim light (area)
    bpy.ops.object.light_add(type='AREA', location=(-5, -5, 5))
    rim_light = bpy.context.view_layer.objects.active
    rim_light.data.energy = 0.5  # Reduced from 2.0
    rim_light.data.size = 3.0  # Smaller area light for rim effect
    
    # Update light positions relative to camera
    main_light.location = camera_location + mathutils.Vector((0, 0, 5))
    main_light.rotation_euler = camera_rotation
    
    # Position fill light opposite to main light
    fill_offset = mathutils.Vector((3, -3, 2))
    fill_light.location = camera_location + fill_offset
    fill_light.rotation_euler = camera_rotation
    
    # Position rim light for edge highlighting
    rim_offset = mathutils.Vector((-2, 2, 1))
    rim_light.location = camera_location + rim_offset
    rim_light.rotation_euler = camera_rotation
    
    return main_light, fill_light, rim_light

def generate_camera_views(center, radius, num_elev=4, num_azim=1):
    """Generate camera views around the object"""
    views = []
    
    # Calculate elevation and azimuth angles
    elev_angles = np.linspace(-50, 80, num_elev)  # Avoid 0 and 90 to skip degenerate top/bottom
    azim_angles = np.linspace(0, 360, num_azim, endpoint=False)
    
    # Calculate FOV
    resolution_x = bpy.context.scene.render.resolution_x
    resolution_y = bpy.context.scene.render.resolution_y
    aspect_ratio = resolution_x / resolution_y
    sensor_width = 36  # mm
    sensor_height = sensor_width / aspect_ratio
    focal_length = 80  # mm
    
    # Compute FOVs
    fov_x = 2 * math.atan(sensor_width / (2 * focal_length))
    fov_y = 2 * math.atan(sensor_height / (2 * focal_length))
    fov = min(fov_x, fov_y)
    
    # Calculate camera distance
    cam_dist = radius / math.sin(fov / 2) * 0.9  # Match original code exactly
    
    for elev_deg in elev_angles:
        elev = math.radians(elev_deg)
        for azim_deg in azim_angles:
            azim = math.radians(azim_deg)
            
            # Calculate camera direction
            cam_dir = mathutils.Vector((
                math.cos(azim) * math.cos(elev),
                math.sin(azim) * math.cos(elev),
                math.sin(elev)
            ))
            
            # Calculate camera position
            location = mathutils.Vector(center) + cam_dir * cam_dist
            
            # Calculate camera rotation
            rot_quat = (mathutils.Vector(center) - location).to_track_quat('-Z', 'Y')
            rotation = rot_quat.to_euler()
            
            # Add view
            views.append({
                "location": location,
                "rotation": rotation,
                "fov": fov,
                "elevation": elev_deg,
                "azimuth": azim_deg
            })
    
    return views 