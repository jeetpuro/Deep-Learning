import os
import cv2

def extract_one_frame_per_video(directory, output_folder, frame_second=1):
    """
    Extract exactly 1 frame from each video in a folder.

    Args:
        directory (str): Path to videos
        output_folder (str): Where to save images
        frame_second (int): Which second of the video to extract (default = 1s)
    """

    os.makedirs(output_folder, exist_ok=True)

    video_files = []

    # Collect videos
    for root, _, files in os.walk(directory):
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_files.append(os.path.join(root, file))

    if not video_files:
        print("No videos found!")
        return

    print(f"Found {len(video_files)} videos")

    for i, video_path in enumerate(video_files):
        try:
            cap = cv2.VideoCapture(video_path)

            if not cap.isOpened():
                print(f"Cannot open: {video_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_id = int(fps * frame_second)

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
            ret, frame = cap.read()

            if ret:
                name = os.path.splitext(os.path.basename(video_path))[0]
                output_path = os.path.join(output_folder, f"{name}.jpg")

                cv2.imwrite(output_path, frame)
                print(f"[{i+1}/{len(video_files)}] Saved: {output_path}")
            else:
                print(f"Could not read frame: {video_path}")

            cap.release()

        except Exception as e:
            print(f"Error with {video_path}: {e}")


if __name__ == "__main__":
    video_directory = "/home/admin1/.cache/kagglehub/datasets/fragdude44/dataset/versions/1"
    output_folder = "/home/admin1/Delete_when_done"

    extract_one_frame_per_video(
        directory=video_directory,
        output_folder=output_folder,
        frame_second=1  # extract frame at 1 second
    )