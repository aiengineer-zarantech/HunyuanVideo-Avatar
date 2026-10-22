import runpod
import os
import subprocess
import requests
import shutil
import logging

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
    logger.info(f"[DEBUG] File exists: {os.path.exists(checkpoint_path)}")
    
    if not os.path.exists(checkpoint_path):
        # List volume contents for debugging
        logger.error("[DEBUG] Checkpoint NOT FOUND. Listing /runpod-volume contents:")
        if os.path.exists("/runpod-volume"):
            for root, dirs, files in os.walk("/runpod-volume"):
                level = root.replace("/runpod-volume", "").count(os.sep)
                if level < 3:  # Limit depth to avoid huge logs
                    indent = "  " * level
                    logger.error(f"{indent}{os.path.basename(root)}/")
                    for file in files[:5]:  # Show first 5 files
                        logger.error(f"{indent}  {file}")
        else:
            logger.error("/runpod-volume does not exist at all!")
        
        return {
            "error": "Checkpoint file not found",
            "expected_path": checkpoint_path,
            "hint": "Ensure model weights are uploaded to /runpod-volume/weights/ckpts/..."
        }

    # Build command for single GPU inference
    cmd = [
        "python", "hymm_sp/sample_gpu_poor.py",
        "--input", image_path,
        "--audio_path", audio_path,
        "--ckpt", checkpoint_path,
        "--sample-n-frames", "129",
        "--image-size", "704",
        "--seed", "128",
        "--cfg-scale", "7.5",
        "--infer-steps", "50",
        "--use-deepcache", "1",
        "--flow-shift-eval-video", "5.0",
        "--save-path", "results",
        "--use-fp8",
        "--infer-min"
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
        if result.stderr:
            logger.warning(f"[STDERR] {result.stderr[-1500:]}")

        # Check for output video (different possible names)
        possible_outputs = [
            "results/avatar_out.mp4",
            "results/output.mp4",
            "results/result.mp4",
        ]
        
        video_path = None
        for p in possible_outputs:
            if os.path.exists(p):
                video_path = p
                logger.info(f"Found output video: {video_path}")
                break
        
        if not video_path:
            # List results folder to help find actual filename
            logger.error("No video found. Listing results/ folder:")
            if os.path.exists("results"):
                for f in os.listdir("results"):
                    logger.error(f"  {f}")
            
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
        # This catches actual script failures
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
