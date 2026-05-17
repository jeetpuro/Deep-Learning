#!/usr/bin/env python3
"""
download_and_preprocess.py
One-click: download Kaggle dataset, extract frames, crop faces, split for training.
"""

import os
import sys
import argparse
import glob
import random
import shutil
from pathlib import Path
from typing import List, Optional

# Video/image libs
import cv2
from PIL import Image
import numpy as np

# Kaggle dataset download
try:
    import kagglehub
except Exception:
    kagglehub = None

# Face detector
try:
    import mediapipe as mp
    mp_face_detection = mp.solutions.face_detection
except Exception as e:
    print("mediapipe import failed:", repr(e))
    mp_face_detection = None

# Thread pool
from concurrent.futures import ThreadPoolExecutor, as_completed

def download_kaggle_dataset_kagglehub(dataset: str, out_dir: str):
    """
    Download Kaggle dataset using kagglehub library.
    dataset format: "owner/dataset-slug" e.g., "xdxd003/ff-c23"
    """
    if kagglehub is None:
        raise RuntimeError("kagglehub not installed. Install with `pip install kagglehub`")
    
    print(f"Downloading {dataset} using kagglehub...")
    os.makedirs(out_dir, exist_ok=True)
    
    try:
        # kagglehub.dataset_download returns the path where dataset was downloaded
        dataset_path = kagglehub.dataset_download(dataset)
        print(f"Dataset downloaded to: {dataset_path}")
        return dataset_path
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        raise

def collect_video_files(
    data_root: str,
    exts=(".mp4", ".avi", ".mkv", ".mov"),
    folder_filter: Optional[str] = "deepfakes,original",
    exclude_filter: Optional[str] = None,
) -> List[str]:
    files = []
    filters = []
    excludes = []
    if folder_filter:
        filters = [part.strip().lower() for part in folder_filter.split(",") if part.strip()]
    if exclude_filter:
        excludes = [part.strip().lower() for part in exclude_filter.split(",") if part.strip()]
    for root, _, filenames in os.walk(data_root):
        for f in filenames:
            if not f.lower().endswith(exts):
                continue
            full_path = os.path.join(root, f)
            if excludes:
                parts = [part.lower() for part in Path(full_path).parts]
                if any(token == part for token in excludes for part in parts):
                    continue
            if filters:
                parts = [part.lower() for part in Path(full_path).parts]
                # Require an exact match on the directory name to avoid matching "DeepFakeDetection" when looking for "Deepfakes"
                if not any(token == part for token in filters for part in parts):
                    continue
            files.append(full_path)
    files.sort()
    return files

def extract_one_frame_per_video(
    directory: str,
    output_folder: str,
    frame_second: int = 1,
    skip_existing: bool = False,
    workers: int = 8,
    folder_filter: Optional[str] = "deepfakes,original",
    exclude_filter: Optional[str] = None,
):
    os.makedirs(output_folder, exist_ok=True)
    video_files = collect_video_files(directory, folder_filter=folder_filter, exclude_filter=exclude_filter)
    if not video_files:
        print("No videos found!")
        return 0

    print(f"Found {len(video_files)} videos — extracting frame at {frame_second}s with {workers} workers")

    def process_video(video_path: str) -> int:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                return 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            frame_id = int(fps * frame_second) if fps > 0 else 0
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                return 0

            base = os.path.splitext(os.path.basename(video_path))[0]
            out = os.path.join(output_folder, f"{base}.jpg")
            if skip_existing and os.path.exists(out):
                return 0

            cv2.imwrite(out, frame)
            return 1
        except Exception:
            return 0

    saved = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futures = [ex.submit(process_video, vp) for vp in video_files]
        for i, fut in enumerate(as_completed(futures), 1):
            saved += fut.result()
            if i % 100 == 0 or i == len(video_files):
                print(f"Processed {i}/{len(video_files)} videos")

    print(f"One-frame extraction complete. Frames saved: {saved}")
    return saved

def extract_frames_and_split(
    directory: str,
    out_root: str,
    train_count: int = 700,
    val_count: int = 300,
    duration_seconds: int = 20,
    frame_interval: int = 2,
    class_hint_keywords: Optional[List[str]] = None,
    skip_existing: bool = False,
    seed: int = 42,
    workers: int = 8,
    folder_filter: Optional[str] = "deepfakes,original",
    exclude_filter: Optional[str] = None,
):
    os.makedirs(out_root, exist_ok=True)
    video_files = collect_video_files(directory, folder_filter=folder_filter, exclude_filter=exclude_filter)
    if not video_files:
        print("No videos found.")
        return 0

    def infer_label(path: str):
        lower = path.lower()
        if "original" in lower or "real" in lower:
            return "original"
        return "deepfake"

    random.seed(seed)
    label_to_videos = {"original": [], "deepfake": []}
    for vf in video_files:
        lbl = infer_label(vf)
        label_to_videos[lbl].append(vf)

    train_videos = []
    val_videos = []
    
    print(f"Found {len(video_files)} total matching videos.")
    for lbl, vids in label_to_videos.items():
        vids.sort() # Sort first for reproducibility
        random.shuffle(vids) # Then shuffle
        
        # Calculate 70% for train, 30% for val dynamically to use ALL data
        t_count = int(len(vids) * 0.7)
        v_count = len(vids) - t_count
        
        t_slice = vids[:t_count]
        v_slice = vids[t_count:t_count + v_count]
        
        train_videos.extend(t_slice)
        val_videos.extend(v_slice)
        print(f"  {lbl}: {len(vids)} total -> {len(t_slice)} train, {len(v_slice)} val")

    print(f"Total Train split: {len(train_videos)} videos.")
    print(f"Total Validation split: {len(val_videos)} videos.")

    def process_split(video_list, split_name):
        def process_video(video_path: str) -> int:
            local_saved = 0
            try:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    return 0
                fps = cap.get(cv2.CAP_PROP_FPS) or 0
                label = infer_label(video_path)
                out_dir = os.path.join(out_root, split_name, label)
                os.makedirs(out_dir, exist_ok=True)

                base = os.path.splitext(os.path.basename(video_path))[0]
                for sec in range(0, duration_seconds, frame_interval):
                    frame_id = int(fps * sec) if fps > 0 else 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    out_name = f"{base}_sec{sec}.jpg"
                    out_path = os.path.join(out_dir, out_name)
                    if skip_existing and os.path.exists(out_path):
                        continue

                    cv2.imwrite(out_path, frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    local_saved += 1

                cap.release()
                return local_saved
            except Exception:
                return 0

        split_saved = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
            futures = [ex.submit(process_video, vp) for vp in video_list]
            for i, fut in enumerate(as_completed(futures), 1):
                split_saved += fut.result()
                if i % 50 == 0 or i == len(video_list):
                    print(f"{split_name}: processed {i}/{len(video_list)} videos")

        print(f"{split_name} extraction complete. Frames saved: {split_saved}")
        return split_saved

    total = 0
    if train_videos:
        total += process_split(train_videos, "Train")
    if val_videos:
        total += process_split(val_videos, "Validation")
    print("Total frames extracted:", total)
    return total

def crop_faces_from_frames(
    frames_root: str,
    out_root: str,
    image_size: int = 224,
    skip_existing: bool = False,
    crop_workers: int = 4,
):
    # Fallback to OpenCV Haar cascade since ctypes is broken for ML libraries
    detector = get_face_detector()

    saved = 0
    tasks = []
    for split in ("Train", "Validation"):
        for label_dir in glob.glob(os.path.join(frames_root, split, "*")):
            if not os.path.isdir(label_dir):
                continue
            label = os.path.basename(label_dir)
            dst_dir = os.path.join(out_root, split, label)
            os.makedirs(dst_dir, exist_ok=True)
            imgs = glob.glob(os.path.join(label_dir, "*.jpg"))
            print(f"Processing {split}/{label}: {len(imgs)} images")
            for p in imgs:
                tasks.append((p, dst_dir))

    def process_image_with_detector(task, local_detector):
        p, dst_dir = task
        try:
            base = os.path.splitext(os.path.basename(p))[0]
            out_path = os.path.join(dst_dir, f"{base}_face.jpg")
            if skip_existing and os.path.exists(out_path):
                return 0

            img = cv2.imread(p)
            if img is None:
                return 0
                
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = local_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(60, 60),
            )
            
            if len(faces) == 0:
                return 0
                
            # Get the largest face
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w_face, h_face = faces[0]
            
            # Add some margin
            margin = 0.2
            xmin = max(0, int(x - w_face * margin))
            ymin = max(0, int(y - h_face * margin))
            xmax = min(img.shape[1], int(x + w_face * (1 + margin)))
            ymax = min(img.shape[0], int(y + h_face * (1 + margin)))
            
            face_crop = img[ymin:ymax, xmin:xmax]
            if face_crop.size == 0:
                return 0
                
            face_resized = cv2.resize(face_crop, (image_size, image_size))
            cv2.imwrite(out_path, face_resized, [cv2.IMWRITE_JPEG_QUALITY, 95])
            return 1
                
        except Exception:
            return 0

    def process_image(task):
        return process_image_with_detector(task, detector)

    if crop_workers <= 1:
        for i, t in enumerate(tasks, 1):
            saved += process_image(t)
            if i % 200 == 0 or i == len(tasks):
                print(f"Cropping progress: {i}/{len(tasks)}")
    else:
        # OpenCV Haar Detectors are notoriously NOT thread-safe in Python.
        # We must instantiate a new detector per thread.
        def process_image_threaded(task):
            t_detector = get_face_detector()
            return process_image_with_detector(task, t_detector)

        with ThreadPoolExecutor(max_workers=max(1, crop_workers)) as ex:
            futures = [ex.submit(process_image_threaded, t) for t in tasks]
            for i, fut in enumerate(as_completed(futures), 1):
                saved += fut.result()
                if i % 200 == 0 or i == len(tasks):
                    print(f"Cropping progress: {i}/{len(tasks)}")

    print(f"Total cropped faces saved: {saved}")
    return saved


def get_face_detector() -> cv2.CascadeClassifier:
    cascade_path = os.path.join(cv2.data.haarcascades, "haarcascade_frontalface_default.xml")
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"Could not load Haar cascade from {cascade_path}")
    return detector


def iter_images(root_dir: str):
    for root, _, files in os.walk(root_dir):
        for file_name in files:
            if file_name.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                yield os.path.join(root, file_name)


def ensure_parent(path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)


def validate_and_sort_faces(source_dir: str, extra_check_dir: str, min_face_area_ratio: float = 0.08):
    detector = get_face_detector()
    total = 0
    moved = 0
    kept = 0
    failed_reads = 0

    source_dir = os.path.abspath(source_dir)
    extra_check_dir = os.path.abspath(extra_check_dir)

    for image_path in iter_images(source_dir):
        total += 1
        image = cv2.imread(image_path)
        if image is None:
            failed_reads += 1
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
        )

        image_area = float(image.shape[0] * image.shape[1])
        best_face_ratio = 0.0
        if len(faces) > 0:
            best_face_ratio = max((w * h) / image_area for (_, _, w, h) in faces)

        if len(faces) == 0 or best_face_ratio < min_face_area_ratio:
            rel_path = os.path.relpath(image_path, source_dir)
            target_path = os.path.join(extra_check_dir, rel_path)
            ensure_parent(target_path)
            shutil.move(image_path, target_path)
            moved += 1
            print(f"MOVED  -> {rel_path}")
        else:
            kept += 1

    print("\nCleanup summary")
    print(f"  scanned      : {total}")
    print(f"  kept         : {kept}")
    print(f"  moved        : {moved}")
    print(f"  unreadable   : {failed_reads}")
    print(f"  extra_check  : {extra_check_dir}")
    return moved


def main():
    parser = argparse.ArgumentParser(description="Download & preprocess dataset (frames + face crops)")
    parser.add_argument("--dataset", default="xdxd003/ff-c23", help="kaggle dataset slug (owner/dataset)")
    parser.add_argument("--data_dir", default="data_raw", help="where kagglehub caches/downloads dataset")
    parser.add_argument("--frames_dir", default="frames", help="where to save extracted frames")
    parser.add_argument("--cropped_dir", default="cropped_faces", help="where to save cropped face images")
    parser.add_argument("--one_frame", action="store_true", help="only extract 1 frame per video (fast)")
    parser.add_argument("--frame_second", type=int, default=1, help="second to sample for one-frame mode")
    parser.add_argument("--train_count", type=int, default=700)
    parser.add_argument("--val_count", type=int, default=300)
    parser.add_argument("--duration_seconds", type=int, default=20)
    parser.add_argument("--frame_interval", type=int, default=2)
    parser.add_argument("--device", default="cuda" if cv2.cuda.getCudaEnabledDeviceCount() > 0 else "cpu")
    parser.add_argument("--skip_download", action="store_true", help="skip download if data already exists")
    parser.add_argument("--skip_existing", action="store_true", help="skip outputs that already exist")
    parser.add_argument("--only_crop", action="store_true", help="skip download/extraction and crop from existing frames")
    parser.add_argument("--only_cleanup", action="store_true", help="skip download/extraction/cropping and just clean cropped faces")
    parser.add_argument("--seed", type=int, default=42, help="random seed for stable train/val split")
    parser.add_argument("--folder_filter", default="deepfakes,original", help="comma-separated exact folder names to include; empty for all")
    parser.add_argument("--exclude_folder", default=None, help="comma-separated exact folder names to exclude; use DeepFakeValidation to skip that dataset")
    parser.add_argument("--workers", type=int, default=min(16, max(1, (os.cpu_count() or 4))), help="workers for video/frame extraction")
    parser.add_argument("--crop_workers", type=int, default=max(1, min(8, (os.cpu_count() or 4) // 2)), help="workers for face cropping on CPU")
    parser.add_argument("--extra_check_dir", default="extra_check", help="where likely false positives are moved after cleanup")
    parser.add_argument("--min_face_area_ratio", type=float, default=0.08, help="minimum detected face area ratio to keep during cleanup")
    args = parser.parse_args()

    folder_filter = args.folder_filter.strip() or None
    exclude_filter = args.exclude_folder.strip() if args.exclude_folder else None
    if exclude_filter:
        exclude_filter = exclude_filter or None

    # 1) Download dataset
    dataset_path = args.data_dir

    if not args.only_crop and not args.only_cleanup:
        if not args.skip_download:
            try:
                dataset_path = download_kaggle_dataset_kagglehub(args.dataset, args.data_dir)
            except Exception as e:
                print(f"Download failed: {e}")
                sys.exit(1)
        else:
            print(f"Skipping download (using existing: {dataset_path})")

        # 2) Extract frames
        print("\n--- FRAME EXTRACTION ---")
        if args.one_frame:
            extract_one_frame_per_video(
                dataset_path,
                args.frames_dir,
                frame_second=args.frame_second,
                skip_existing=args.skip_existing,
                workers=args.workers,
                folder_filter=folder_filter,
                exclude_filter=exclude_filter,
            )
        else:
            extract_frames_and_split(
                dataset_path,
                args.frames_dir,
                train_count=args.train_count,
                val_count=args.val_count,
                duration_seconds=args.duration_seconds,
                frame_interval=args.frame_interval,
                skip_existing=args.skip_existing,
                seed=args.seed,
                workers=args.workers,
                folder_filter=folder_filter,
                exclude_filter=exclude_filter,
            )
    else:
        print("Skipping download/extraction (--only_crop/--only_cleanup).")

    # 3) Crop faces from frames + save to cropped_dir
    if not args.only_cleanup:
        print("\n--- FACE CROPPING ---")
        os.makedirs(args.cropped_dir, exist_ok=True)
        crop_faces_from_frames(
            args.frames_dir,
            args.cropped_dir,
            image_size=224,
            skip_existing=args.skip_existing,
            crop_workers=args.crop_workers,
        )

    if not args.only_crop:
        print("\n--- CLEANUP PASS ---")
        os.makedirs(args.extra_check_dir, exist_ok=True)
        validate_and_sort_faces(
            source_dir=args.cropped_dir,
            extra_check_dir=args.extra_check_dir,
            min_face_area_ratio=args.min_face_area_ratio,
        )

    print(f"\n✓ Done! Cropped faces ready for training in: {args.cropped_dir}")

if __name__ == "__main__":
    main()