import os
import json
import numpy as np
from datetime import datetime

def ensure_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)

def save_metadata(metadata_path, data):
    """Save metadata to JSON file"""
    with open(metadata_path, 'w') as f:
        json.dump(data, f, indent=4)

def generate_metadata(model_name, num_views, resolution, fov):
    """Generate metadata dictionary"""
    return {
        "model_name": model_name,
        "num_views": num_views,
        "resolution": resolution,
        "fov": fov,
        "timestamp": datetime.now().isoformat(),
        "version": "1.0"
    }

def format_matrix(matrix):
    """Format matrix for printing"""
    return "\n".join([" ".join([f"{v:8.4f}" for v in row]) for row in matrix])

def print_matrix(name, matrix):
    """Print matrix with name"""
    print(f"\n{name}:")
    print(format_matrix(matrix)) 