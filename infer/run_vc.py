# run_vc.py
import os
import sys
import subprocess

# Step 0: Ensure models folder exists
os.makedirs("models", exist_ok=True)

# Step 1: Download index and model from Drive using gdown
def download_models():
    import gdown

    index_url = "https://drive.google.com/uc?id=1gpk10q7DdwGgwa3uwoA_23kdpUFlE74W"
    pth_url = "https://drive.google.com/uc?id=1RekGRj8oX5wQB5rLXWecN-XPol6FpXpT"

    index_path = "models/index.npy"
    model_path = "models/model.pth"

    if not os.path.exists(index_path):
        print("Downloading index.npy...")
        gdown.download(index_url, index_path, quiet=False)
    else:
        print("index.npy already exists.")

    if not os.path.exists(model_path):
        print("Downloading model.pth...")
        gdown.download(pth_url, model_path, quiet=False)
    else:
        print("model.pth already exists.")

# Step 2: Install required packages
def install_dependencies():
    packages = [
        "numpy<2",
        "torch==2.2.0",
        "torchaudio==2.2.0",
        "pyworld",
        "praat-parselmouth",
        "faiss-cpu",
        "gdown"
    ]
    subprocess.check_call([sys.executable, "-m", "pip", "install", *packages])

# Step 3: Import the pipeline
def import_pipeline():
    sys.path.append(os.path.join(os.getcwd(), "infer"))
    from modules.vc.pipeline import Pipeline
    return Pipeline

# Step 4: Run voice conversion
def run_vc(input_audio, output_audio):
    Pipeline = import_pipeline()

    # Config: update according to your pipeline defaults
    import json
    config_path = "configs/config.json"
    import json
    with open(config_path, "r") as f:
        cfg_dict = json.load(f)

    # Initialize pipeline
    pipeline = Pipeline(tgt_sr=22050, config=cfg_dict)

    # Run conversion
    print(f"Converting {input_audio} -> {output_audio}...")
    pipeline.pipeline(input_audio_path=input_audio, output_path=output_audio)
    print("Conversion done!")

# ---------------- Main ----------------
if __name__ == "__main__":
    # Install dependencies
    install_dependencies()

    # Download models
    download_models()

    # Example usage (replace with your file names)
    input_audio = "Vocals.mp3"  # put your input audio here
    output_audio = "output.wav"
    run_vc(input_audio, output_audio)
