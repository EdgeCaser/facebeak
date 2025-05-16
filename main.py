import os
import argparse
import sys
from detection import detect_crows
from tracking import assign_crow_ids
from utils import extract_frames, save_video_with_labels

class TeeLogger:
    def __init__(self, log_path):
        self.log = open(log_path, 'w')
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        sys.stdout = self
        sys.stderr = self
    def write(self, data):
        self.log.write(data)
        self.log.flush()
        self.stdout.write(data)
        self.stdout.flush()
    def flush(self):
        self.log.flush()
        self.stdout.flush()
    def close(self):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        self.log.close()

def main(video_path, output_path, frame_skip=5, detection_threshold=0.5, similarity_threshold=0.85):
    print("[INFO] Extracting frames...")
    frames = extract_frames(video_path, skip=frame_skip)

    print("[INFO] Running crow detection...")
    detections = detect_crows(frames, score_threshold=detection_threshold)

    print("[INFO] Assigning crow identities...")
    labeled_frames = assign_crow_ids(frames, detections)

    print(f"[INFO] Saving output to {output_path}...")
    save_video_with_labels(labeled_frames, output_path)
    print("[DONE] Video saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, help="Path to input video")
    parser.add_argument("--output", required=True, help="Path to save output video")
    parser.add_argument("--skip", type=int, default=5, help="Frame skip interval")
    parser.add_argument("--detection-threshold", type=float, default=0.5, help="Detection confidence threshold (default: 0.5)")
    parser.add_argument("--similarity-threshold", type=float, default=0.85, help="Crow ID similarity threshold (default: 0.85)")
    args = parser.parse_args()

    # Overwrite log at start of each run
    log_path = os.path.join(os.path.dirname(__file__), 'facebeak_session.log')
    logger = TeeLogger(log_path)
    try:
        print(f"[LOG] Starting facebeak session. Video: {args.video}, Output: {args.output}, Skip: {args.skip}, Detection threshold: {args.detection_threshold}")
        frames = extract_frames(args.video, skip=args.skip)
        print(f"[LOG] Extracted {len(frames)} frames.")
        detections = detect_crows(frames, score_threshold=args.detection_threshold)
        print(f"[LOG] Detection complete.")
        labeled_frames = assign_crow_ids(frames, detections)
        print(f"[LOG] Tracking complete.")
        save_video_with_labels(labeled_frames, args.output)
        print(f"[LOG] Output video saved to {args.output}.")
        print("[DONE] Video saved.")
    except Exception as e:
        print(f"[ERROR] {e}")
        raise
    finally:
        logger.close()
