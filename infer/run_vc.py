import os
import sys
import argparse
import numpy as np
import gdown
import torch

# Add your repo to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import Pipeline
from infer.modules.vc.pipeline import Pipeline

# --------------------------
# Index wrapper for Pipeline
# --------------------------
class IndexWrapper:
    """
    Wraps a NumPy dict/array into an object with attributes
    expected by Pipeline.
    """
    def __init__(self, index_dict):
        self.x_pad = index_dict.get("x_pad", 0)
        self.x_query = index_dict.get("x_query", 0)
        self.x_center = index_dict.get("x_center", 0)
        self.x_max = index_dict.get("x_max", 0)
        self.is_half = index_dict.get("is_half", False)
        self.index_data = index_dict

# --------------------------
# Download model/index if not present
# --------------------------
INDEX_URL = "https://drive.google.com/uc?id=1gpk10q7DdwGgwa3uwoA_23kdpUFlE74W"
MODEL_URL = "https://drive.google.com/uc?id=1RekGRj8oX5wQB5rLXWecN-XPol6FpXpT"

os.makedirs("models", exist_ok=True)
index_path = "models/index.npy"
model_path = "models/model.pth"

if not os.path.exists(index_path):
    gdown.download(INDEX_URL, index_path, quiet=False)

if not os.path.exists(model_path):
    gdown.download(MODEL_URL, model_path, quiet=False)

# --------------------------
# Argument parser
# --------------------------
parser = argparse.ArgumentParser(description="Run voice conversion")
parser.add_argument("--input", type=str, required=True, help="Input audio path")
parser.add_argument("--output", type=str, default="output.wav", help="Output path")
parser.add_argument("--tgt_sr", type=int, default=22050, help="Target sample rate")
args = parser.parse_args()

# --------------------------
# Load index
# --------------------------
index = np.load(index_path, allow_pickle=True).item()
wrapped_index = IndexWrapper(index)

# --------------------------
# Initialize pipeline
# --------------------------
pipeline = Pipeline(tgt_sr=args.tgt_sr, config=wrapped_index)

# --------------------------
# Run conversion
# --------------------------
pipeline.vc(input_audio_path=args.input, output_audio_path=args.output,
            model_path=model_path)

print(f"Voice conversion done! Output saved at {args.output}")
