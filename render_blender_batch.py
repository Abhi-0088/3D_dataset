import subprocess
import sys
import os

# Path to Blender executable (update this if needed)
BLENDER_PATH = r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe"  # Or full path, e.g. r"C:\Program Files\Blender Foundation\Blender 3.6\blender.exe"

# Path to your Blender Python script
generate_views_script = os.path.abspath("generate_views.py")

# Build the command
cmd = [
    BLENDER_PATH,
    "--background",
    "--python", generate_views_script
]

# Run the command
try:
    subprocess.run(cmd, check=True)
except subprocess.CalledProcessError as e:
    print(f"Blender batch render failed: {e}")
    sys.exit(1) 