import bpy

def create_camera_and_lights():
    """Create camera and lights if they don't exist"""
    scene = bpy.context.scene
    
    # Create camera
    bpy.ops.object.camera_add(location=(0, 0, 0))
    camera = bpy.context.active_object
    camera.name = "Camera"
    scene.camera = camera
    
    # Create lights
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 0))
    main_light = bpy.context.active_object
    main_light.name = "MainLight"
    
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 0))
    fill_light = bpy.context.active_object
    fill_light.name = "FillLight"
    
    bpy.ops.object.light_add(type='SUN', location=(0, 0, 0))
    rim_light = bpy.context.active_object
    rim_light.name = "RimLight"
    
    # Set render settings
    scene.render.engine = 'CYCLES'
    scene.cycles.device = 'GPU'
    scene.render.film_transparent = True
    
    # Verify camera exists
    if "Camera" not in bpy.data.objects:
        raise RuntimeError("Failed to create camera")
    
    # Verify lights exist
    if "MainLight" not in bpy.data.objects or "FillLight" not in bpy.data.objects or "RimLight" not in bpy.data.objects:
        raise RuntimeError("Failed to create lights")
    
    return camera, main_light, fill_light, rim_light 