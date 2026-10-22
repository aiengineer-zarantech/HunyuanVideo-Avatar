import runpod
import os
import subprocess
import requests
import shutil
import logging
import csv

# Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def download_file(url, save_path, timeout=60):
    """Download file from URL"""
    try:
        logger.info(f"Downloading {url} to {save_path}")
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)
        logger.info(f"Download complete: {save_path}")
        return True
    except Exception as e:
        logger.error(f"[Download Error] {url} -> {e}")
        return False

def handler(event):
    """RunPod Serverless main handler"""
    data = event.get("input", {})
    image_url = data.get("image_url")
    audio_url = data.get("audio_url")

    # Validate input
    if not image_url or not audio_url:
        return {"error": "Missing image_url or audio_url"}

    # Create directories
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    image_path = "uploads/input_image.png"
    audio_path = "uploads/input_audio.wav"

    # Download files
    if not download_file(image_url, image_path):
        return {"error": "Failed to download image"}
    if not download_file(audio_url, audio_path):
        return {"error": "Failed to download audio"}

    # Environment setup
    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace"
    env["HF_HOME"] = "/workspace/.cache/huggingface"
    env["TORCH_HOME"] = "/workspace/.cache/torch"
    env["TRANSFORMERS_OFFLINE"] = "0"
    env["DISABLE_SP"] = "1"  # Single GPU mode

    # ✅ CORRECT PATH: Global Volume mounts at /runpod-volume
    checkpoint_path = "/runpod-volume/weights/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt"
    
    # Debug: Check if checkpoint exists
    logger.info(f"[DEBUG] Checking checkpoint at: {checkpoint_path}")
    if not os.path.exists(checkpoint_path):
        logger.error("[DEBUG] Checkpoint NOT FOUND.")
        return {"error": "Checkpoint file not found", "expected_path": checkpoint_path}
    else:
        logger.info("[DEBUG] File exists: True")

    # ✅ CREATE CSV for sample_gpu_poor.py
    csv_path = "uploads/input.csv"
    with open(csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Header (sample_gpu_poor.py is format ko expect karta hai)
        writer.writerow(['image_path', 'audio_path', 'name'])
        writer.writerow([image_path, audio_path, 'my_avatar'])
        
    vae_path = "/runpod-volume/weights/ckpts/hunyuan-video-t2v-720p/vae"
    logger.info(f"Created CSV at {csv_path}")

    # Build command for single GPU inference using CSV
    cmd = [
        "python", "hymm_sp/sample_gpu_poor.py",
        "--input", csv_path,  # <-- CSV File pass ki
        "--ckpt", checkpoint_path,
        "--vae", vae_path,
        "--sample-n-frames", "129",
        "--image-size", "704",
        "--seed", "128",
        "--cfg-scale", "7.5",
        "--infer-steps", "50",
        "--use-deepcache", "1",
        "--flow-shift-eval-video", "5.0",
        "--save-path", "results",
        "--use-fp8",
        "--infer-min",
    ]

    logger.info(f"[INFO] Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd="/workspace",
            env=env
        )
        
        logger.info(f"[STDOUT] {result.stdout[-1500:]}")

        # Check for output video
        possible_outputs = [
            "results/avatar_out.mp4",
            "results/output.mp4",
            "results/result.mp4",
        ]
        
        video_path = None
        for p in possible_outputs:
            if os.path.exists(p):
                video_path = p
                break
        
        # Agar upar wale naam se nahi mili, toh results folder me koi bhi mp4 dhundhein
        if not video_path and os.path.exists("results"):
            for f in os.listdir("results"):
                if f.endswith(".mp4"):
                    video_path = os.path.join("results", f)
                    break

        if not video_path:
            return {
                "status": "failed",
                "message": "Script completed but no video file found",
                "stdout": result.stdout[-2000:],
                "stderr": result.stderr[-2000:] if result.stderr else "No stderr",
            }
        
        return {
            "status": "success",
            "video_path": video_path,
            "stdout": result.stdout[-500:],
        }

    except subprocess.CalledProcessError as e:
        logger.error(f"Inference script crashed with code {e.returncode}")
        return {
            "error": "Inference script crashed",
            "return_code": e.returncode,
            "stdout": e.stdout[-2000:] if e.stdout else "No stdout",
            "stderr": e.stderr[-2000:] if e.stderr else "No stderr",
        }
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return {"error": f"Unexpected error: {str(e)}"}

# Start RunPod Serverless
runpod.serverless.start({"handler": handler})
