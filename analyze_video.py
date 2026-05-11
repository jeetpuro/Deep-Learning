import os
import cv2
import sys
import statistics

def analyze_videos(directory):
    """
    Analyzes all .mp4 videos in a directory to find the lowest/highest FPS,
    shortest duration, and lowest/highest resolution.

    Args:
        directory (str): The path to the directory containing the video files.
    """
    video_files = []
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".mp4"):
                video_files.append(os.path.join(root, file))

    if not video_files:
        print(f"No .mp4 files found in the directory: {directory}")
        return

    lowest_fps = (float('inf'), None)
    highest_fps = (0, None)
    shortest_duration = (float('inf'), None)
    highest_duration = (0, None)
    lowest_resolution = (float('inf'), None)
    highest_resolution = (0, None)
    
    total_duration = 0
    video_count = 0
    all_durations = []

    print(f"Analyzing {len(video_files)} video files...")

    for video_path in video_files:
        try:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error opening video file: {video_path}")
                continue

            fps = cap.get(cv2.CAP_PROP_FPS)
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            
            if fps > 0:
                duration = frame_count / fps
            else:
                duration = 0

            resolution = width * height
            
            total_duration += duration
            video_count += 1
            all_durations.append(duration)

            if fps < lowest_fps[0]:
                lowest_fps = (fps, video_path)
            if fps > highest_fps[0]:
                highest_fps = (fps, video_path)
            if duration < shortest_duration[0]:
                shortest_duration = (duration, video_path)
            if duration > highest_duration[0]:
                highest_duration = (duration, video_path)
            if resolution < lowest_resolution[0]:
                lowest_resolution = (resolution, video_path)
            if resolution > highest_resolution[0]:
                highest_resolution = (resolution, video_path)

            cap.release()
        except Exception as e:
            print(f"Could not process {video_path}: {e}")

    print("\n--- Video Analysis Results ---")
    if video_count > 0:
        avg_duration = total_duration / video_count
        median_duration = statistics.median(all_durations)
        print(f"Average Duration: {avg_duration:.2f} seconds across {video_count} videos")
        print(f"Median Duration: {median_duration:.2f} seconds")
        
    if lowest_fps[1]:
        print(f"Lowest FPS: {lowest_fps[0]:.2f} FPS -> {os.path.basename(lowest_fps[1])}")
    if highest_fps[1]:
        print(f"Highest FPS: {highest_fps[0]:.2f} FPS -> {os.path.basename(highest_fps[1])}")
    if shortest_duration[1]:
        print(f"Shortest Duration: {shortest_duration[0]:.2f} seconds -> {os.path.basename(shortest_duration[1])}")
    if highest_duration[1]:
        print(f"Highest Duration: {highest_duration[0]:.2f} seconds -> {os.path.basename(highest_duration[1])}")
    
    if lowest_resolution[1]:
        cap = cv2.VideoCapture(lowest_resolution[1])
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Lowest Resolution: {width}x{height} -> {os.path.basename(lowest_resolution[1])}")
        cap.release()

    if highest_resolution[1]:
        cap = cv2.VideoCapture(highest_resolution[1])
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"Highest Resolution: {width}x{height} -> {os.path.basename(highest_resolution[1])}")
        cap.release()
    print("----------------------------")

    if all_durations:
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(10, 6))
            # Create a histogram to show the distribution of durations
            plt.hist(all_durations, bins=15, color='skyblue', edgecolor='black')
            
            # Add vertical lines for average and median to compare them
            plt.axvline(avg_duration, color='green', linestyle='dashed', linewidth=2, label=f'Average: {avg_duration:.2f}s')
            plt.axvline(median_duration, color='red', linestyle='dashed', linewidth=2, label=f'Median: {median_duration:.2f}s')
            
            plt.title('Distribution of Video Durations')
            plt.xlabel('Duration (seconds)')
            plt.ylabel('Number of Videos')
            plt.legend()
            
            print("\nShowing duration distribution graph...")
            plt.show()
        except ImportError:
            print("\nNote: The 'matplotlib' library is required to plot the graph.")
            print("You can install it by running: pip install matplotlib")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        video_directory = sys.argv[1]
        if os.path.isdir(video_directory):
            analyze_videos(video_directory)
        else:
            print(f"Error: The provided path '{video_directory}' is not a valid directory.")
    else:
        # As a fallback, check the subdirectories in the current workspace
        workspace_dirs = [d for d in os.listdir('.') if os.path.isdir(d) and not d.startswith('.')]
        if workspace_dirs:
            print("No directory specified. Please choose a directory to analyze:")
            for i, dirname in enumerate(workspace_dirs):
                print(f"{i + 1}: {dirname}")
            
            try:
                choice = int(input("Enter the number of the directory: ")) - 1
                if 0 <= choice < len(workspace_dirs):
                    selected_dir = workspace_dirs[choice]
                    print(f"Analyzing videos in: {selected_dir}")
                    analyze_videos(selected_dir)
                else:
                    print("Invalid choice.")
            except ValueError:
                print("Invalid input. Please enter a number.")
        else:
            print("Usage: python analyze_videos.py <path_to_video_directory>")
            print("Or run in a directory with subdirectories containing videos.")
