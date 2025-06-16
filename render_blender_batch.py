import subprocess
import sys
import os

# Path to Blender executable (update this if needed)
BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe"

# Get the absolute path of the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Path to your Blender Python script
generate_views_script = os.path.join(current_dir, "generate_views.py")

# Build the command
cmd = [
    BLENDER_PATH,
    "--background",
    "--python", generate_views_script
]

# Run the command
try:
    # Run Blender with the current directory as working directory
    subprocess.run(cmd, check=True, cwd=current_dir)
except subprocess.CalledProcessError as e:
    print(f"Blender batch render failed: {e}")
    sys.exit(1) 