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

    # ✅ 修正環境：確保 hymm_sp 模組與 flash_attn dummy 可被找到
    env = os.environ.copy()
    env["PYTHONPATH"] = "/workspace"
    env["HF_HOME"] = "/workspace/.cache/huggingface"
    env["TORCH_HOME"] = "/workspace/.cache/torch"
    env["TRANSFORMERS_OFFLINE"] = "0"  # 若需要離線部署可改成 "1"

    # ▶️ 執行 HunyuanVideo-Avatar 推理
    cmd = [
        "python", "hymm_sp/inference.py",
        "--config", "configs/hunyuan_avatar.yaml",
        "--image_path", image_path,
        "--audio_path", audio_path,
        "--output_dir", "results"
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

        video_path = "results/avatar_out.mp4"
        if os.path.exists(video_path):
            return {
                "status": "success",
                "video_path": video_path,
                "stdout": result.stdout[-800:],  # 顯示最後部分的log
            }
        else:
            return {
                "status": "failed",
                "message": "No video generated",
                "stdout": result.stdout[-800:],
                "stderr": result.stderr[-800:],
            }

    except subprocess.CalledProcessError as e:
        return {
            "error": "Inference script crashed",
            "stdout": e.stdout[-2000:] if e.stdout else None,
            "stderr": e.stderr[-2000:] if e.stderr else None,
        }


# ✅ RunPod Serverless 啟動點
runpod.serverless.start({"handler": handler})
