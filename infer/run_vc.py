import argparse
import os
import sys
import gdown
import numpy as np
import torch

# Ensure repo root is in PYTHONPATH
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from modules.vc.pipeline import Pipeline

def download_drive_file(url, output_path):
    """Download file from Google Drive using gdown if it doesn't exist."""
    if not os.path.exists(output_path):
        print(f"Downloading {output_path} ...")
        gdown.download(url, output_path, quiet=False)
    else:
        print(f"{output_path} already exists, skipping download.")

def main():
    parser = argparse.ArgumentParser(description="RVC Voice Conversion")
    parser.add_argument("--input", required=True, help="Path to input audio")
    parser.add_argument("--output", required=True, help="Path to output audio")
    parser.add_argument("--index", required=True, help="Index file path or Drive URL")
    parser.add_argument("--model", required=True, help="Model file path or Drive URL")
    parser.add_argument("--tgt_sr", default=22050, type=int, help="Target sampling rate")
    args = parser.parse_args()

    # Download index/model if they are URLs
    if args.index.startswith("http"):
        download_drive_file(args.index, "index.npy")
        index_path = "index.npy"
    else:
        index_path = args.index

    if args.model.startswith("http"):
        download_drive_file(args.model, "model.pth")
        model_path = "model.pth"
    else:
        model_path = args.model

    # Load index and model
    print("Loading index and model...")
    index = np.load(index_path, allow_pickle=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    pipeline = Pipeline(tgt_sr=args.tgt_sr, config=index)

    # Run voice conversion
    print(f"Processing {args.input} ...")
    pipeline.vc(args.input, args.output)
    print(f"Output saved to {args.output}")

if __name__ == "__main__":
    main()
