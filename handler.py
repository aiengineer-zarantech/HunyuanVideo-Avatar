import runpod
import os
import subprocess
import requests
import shutil


def download_file(url, save_path, timeout=30):
    """下載檔案並確保成功"""
    try:
        r = requests.get(url, stream=True, timeout=timeout)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            shutil.copyfileobj(r.raw, f)
        return True
    except Exception as e:
        print(f"[Download Error] {url} -> {e}")
        return False


def handler(event):
    """RunPod Serverless 主入口"""
    data = event.get("input", {})
    image_url = data.get("image_url")
    audio_url = data.get("audio_url")

    # 🧩 驗證輸入
    if not image_url or not audio_url:
        return {"error": "Missing image_url or audio_url"}

    # 📁 建立資料夾
    os.makedirs("uploads", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    image_path = "uploads/input_image.png"
    audio_path = "uploads/input_audio.wav"

    # 🌐 下載圖片與音訊
    if not download_file(image_url, image_path):
        return {"error": "Failed to download image"}
    if not download_file(audio_url, audio_path):
        return {"error": "Failed to download audio"}

    # ✅ 環境設定
    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace"
    env["HF_HOME"] = "/workspace/.cache/huggingface"
    env["TORCH_HOME"] = "/workspace/.cache/torch"
    env["TRANSFORMERS_OFFLINE"] = "0"
    env["DISABLE_SP"] = "1"   # single-GPU 必須

    # ▶️ 執行 HunyuanVideo-Avatar 單卡推理 (sample_gpu_poor.py)
    checkpoint_path = "./weights/ckpts/hunyuan-video-t2v-720p/transformers/mp_rank_00_model_states_fp8.pt"
    
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

    print(f"[INFO] Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=True,
            text=True,
            cwd="/workspace",
            env=env
        )

        # sample_gpu_poor.py 的輸出檔名可能不同，這裡列出常見可能
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

        if video_path:
            return {
                "status": "success",
                "video_path": video_path,
                "stdout": result.stdout[-800:],
            }
        else:
            return {
                "status": "failed",
                "message": "Script ran but no video file found",
                "stdout": result.stdout[-2000:],
                "stderr": result.stderr[-2000:],
            }

    except subprocess.CalledProcessError as e:
        # 這裡會捕捉到真正的錯誤，方便除錯
        return {
            "error": "Inference script crashed",
            "stdout": e.stdout[-2000:] if e.stdout else None,
            "stderr": e.stderr[-2000:] if e.stderr else None,
        }


# ✅ RunPod Serverless 啟動點
runpod.serverless.start({"handler": handler})
