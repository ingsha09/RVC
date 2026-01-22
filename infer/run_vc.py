import argparse
import numpy as np
import torch
from infer.modules.vc.pipeline import Pipeline
from configs.config import Config  # or import your config object

def main():
    parser = argparse.ArgumentParser(description="Run voice conversion")
    parser.add_argument('--input', type=str, required=True, help="Path to input audio")
    parser.add_argument('--output', type=str, default='output.wav', help="Path to save converted audio")
    parser.add_argument('--index', type=str, required=True, help="Path to index.npy")
    parser.add_argument('--model', type=str, required=True, help="Path to model.pth")
    args = parser.parse_args()

    # Load index and model
    index = np.load(args.index, allow_pickle=True)
    model_state = torch.load(args.model, map_location='cpu')

    # Load config
    cfg = Config()  # If your Config class is parameterless
    pipeline = Pipeline(tgt_sr=22050, config=cfg)
    pipeline.load_model_state(model_state)  # Or whatever your pipeline uses

    # Run conversion
    pipeline.vc(args.input, args.output, index=index)

    print(f"Saved converted audio to {args.output}")

if __name__ == "__main__":
    main()
