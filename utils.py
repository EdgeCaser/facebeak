import cv2
from tqdm import tqdm

def extract_frames(video_path, skip=5):
    cap = cv2.VideoCapture(video_path)
    # Get total frame count for progress bar
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    
    print(f"[INFO] Total frames in video: {total_frames}")
    print(f"[INFO] Processing every {skip}th frame...")
    
    with tqdm(total=total_frames, desc="Extracting frames") as pbar:
        i = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if i % skip == 0:
                frames.append(frame)
            i += 1
            pbar.update(1)
    
    cap.release()
    print(f"[INFO] Extracted {len(frames)} frames")
    return frames

def save_video_with_labels(frames, output_path, fps=15):
    if not frames:
        print("[WARN] No frames to save!")
        return

    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"[INFO] Saving {len(frames)} frames to {output_path}")
    for frame in tqdm(frames, desc="Saving video"):
        out.write(frame)
    out.release()
    print("[INFO] Video save complete")
