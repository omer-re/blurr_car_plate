import subprocess
import cv2
import numpy as np
from tqdm import tqdm
from collections import deque

# Temporal smoothing constants
TEMPORAL_SMOOTHING_WEIGHT = 1  # Adjust this between 0.0 (no smoothing) to 1.0 (high smoothing)


def run_ffmpeg(command):
    """Helper function to run an FFmpeg command."""
    subprocess.run(command, shell=True, check=True)


def downscale_video(input_video, downscaled_video, scale_factor=0.5):
    """Downscale the video using FFmpeg."""
    print("Downscaling video...")
    run_ffmpeg(f"ffmpeg -i {input_video} -vf scale=iw*{scale_factor}:ih*{scale_factor} {downscaled_video}")


def quantize_colors(frame, k=8):
    """Reduces the number of colors in the image using k-means clustering."""
    pixel_values = frame.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    _, labels, centers = cv2.kmeans(
        pixel_values, k, None,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS
    )

    centers = np.uint8(centers)
    quantized_frame = centers[labels.flatten()]
    return quantized_frame.reshape(frame.shape)


def quantize_video(input_video, quantized_video, k=8):
    """Quantize the downscaled video and save the result."""
    cap = cv2.VideoCapture(input_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(quantized_video, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Quantizing Video")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Apply color quantization
        quantized_frame = quantize_colors(frame, k=k)
        out.write(quantized_frame)

        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()


def remove_isolated_segments(mask, min_size=500):
    """Remove small isolated segments from the mask."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
    clean_mask = np.zeros_like(mask)

    # Iterate through connected components and keep only large segments
    for i in range(1, num_labels):  # Skip background
        if stats[i, cv2.CC_STAT_AREA] >= min_size:
            clean_mask[labels == i] = 255

    return clean_mask


def create_mask_from_quantized(quantized_video, mask_video, original_resolution):
    """Create a mask for yellow areas from the quantized video."""
    cap = cv2.VideoCapture(quantized_video)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = original_resolution
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(mask_video, fourcc, fps, (width, height))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Creating Mask")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to HSL and create mask for yellow
        hsl_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        mask = cv2.inRange(hsl_frame, lower_yellow, upper_yellow)

        # Remove isolated segments
        mask = remove_isolated_segments(mask)

        # Upscale mask to original resolution
        mask = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST)
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)  # Convert to BGR to save as video
        out.write(mask_bgr)

        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()


WINDOW_SIZE = 10  # Number of frames for smoothing

def smooth_mask_sliding_window(current_mask, mask_history):
    """
    Smooth the current mask using a sliding window approach.
    Args:
        current_mask (numpy.ndarray): The current binary mask.
        mask_history (deque): A queue containing the last `N` masks.
    Returns:
        numpy.ndarray: The smoothed mask.
    """
    mask_history.append(current_mask.astype(np.float32))
    if len(mask_history) > WINDOW_SIZE:
        mask_history.popleft()
    # Compute the average of the masks in the history
    smoothed_mask = np.mean(mask_history, axis=0)
    return smoothed_mask, mask_history


def smooth_mask(current_mask, previous_mask):
    """
    Smooth the current mask using a weighted average with the previous mask.
    Args:
        current_mask (numpy.ndarray): The current binary mask.
        previous_mask (numpy.ndarray): The previous smoothed mask.
    Returns:
        numpy.ndarray: The smoothed mask.
    """
    # Ensure both masks have the same data type
    current_mask = current_mask.astype(np.float32)
    previous_mask = previous_mask.astype(np.float32)

    smoothed_mask = cv2.addWeighted(
        current_mask,
        TEMPORAL_SMOOTHING_WEIGHT,
        previous_mask,
        1 - TEMPORAL_SMOOTHING_WEIGHT,
        0
    )

    # Convert the smoothed mask back to the original type (if needed)
    return smoothed_mask.astype(np.uint8)


def blur_license_plate_with_polygon_and_smoothing(input_video, mask_video, output_video):
    """Blur license plates using the mask and fill them with forced 4-point polygons, with temporal smoothing."""
    cap_original = cv2.VideoCapture(input_video)
    cap_mask = cv2.VideoCapture(mask_video)
    previous_mask = None  # Initialize previous_mask
    # Verify input video properties
    fps = int(cap_original.get(cv2.CAP_PROP_FPS))
    width = int(cap_original.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap_original.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Input video properties - FPS: {fps}, Width: {width}, Height: {height}")

    if fps == 0 or width == 0 or height == 0:
        print("Error: Invalid video properties. Cannot proceed.")
        return

    # Initialize VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, (width, height))
    if not out.isOpened():
        print("Error: VideoWriter initialization failed!")
        return

    total_frames = int(cap_original.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Filling Mask with Temporal Smoothing")

    # Initialize variables for temporal smoothing
    mask_history = deque(maxlen=WINDOW_SIZE)

    while cap_original.isOpened() and cap_mask.isOpened():
        ret_orig, frame_original = cap_original.read()
        ret_mask, frame_mask = cap_mask.read()
        if not ret_orig or not ret_mask:
            print("Error: Could not read frames from input or mask videos.")
            break

        print("Processing frame...")

        # Convert mask to grayscale
        mask_gray = cv2.cvtColor(frame_mask, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(mask_gray, 1, 255, cv2.THRESH_BINARY)

        # Apply sliding window smoothing
        smoothed_mask, mask_history = smooth_mask_sliding_window(binary_mask, mask_history)
        smoothed_mask = (smoothed_mask > 127).astype(np.uint8) * 255  # Threshold for binary output

        # Apply additional smoothing using cv2.addWeighted
        if previous_mask is not None:
            smoothed_mask = cv2.addWeighted(
                smoothed_mask, TEMPORAL_SMOOTHING_WEIGHT,
                previous_mask, 1 - TEMPORAL_SMOOTHING_WEIGHT,
                0
            )

        # Update the previous_mask for the next frame
        previous_mask = smoothed_mask

        # Find contours
        contours, _ = cv2.findContours(smoothed_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            # Get the minimum area rectangle
            rect = cv2.minAreaRect(contour)
            box = cv2.boxPoints(rect)  # Get the 4 points of the rectangle
            box = np.int0(box)  # Convert to integer coordinates

            # Expand the bounding box slightly
            expansion_ratio = 0.1  # 10% expansion
            box_center = np.mean(box, axis=0)
            box = np.array([(point - box_center) * (1 + expansion_ratio) + box_center for point in box], dtype=np.int0)

            # Draw the 4-point polygon and fill with pink
            cv2.fillPoly(frame_original, [box], (0, 180, 238))  # RGB(238, 180, 0) as BGR in OpenCV

        out.write(frame_original)
        pbar.update(1)

    pbar.close()
    cap_original.release()
    cap_mask.release()
    out.release()
    print("Processing complete.")



def main():
    input_path = r"G:\My Drive\OMER_PERSONAL\PycharmProjects\blurr_car_plate\car_video_no_audio.mp4"
    downscaled_path = r"downscaled.mp4"
    quantized_path = r"quantized.mp4"
    mask_path = r"mask.mp4"
    output_path = r"output_with_smoothing.mp4"

    # Verify input file exists
    import os
    if not os.path.exists(input_path):
        print(f"Error: File not found at {input_path}")
        return

    # Step 1: Downscale the video
    # downscale_video(input_path, downscaled_path, scale_factor=0.5)

    # Step 2: Quantize the downscaled video
    # quantize_video(downscaled_path, quantized_path, k=32)

    # Step 3: Create a mask from the quantized video
    original_resolution = (1920, 1080)  # Replace with your original resolution
    create_mask_from_quantized(quantized_path, mask_path, original_resolution)

    # Step 4: Blur license plates using polygons with smoothing
    blur_license_plate_with_polygon_and_smoothing(input_path, mask_path, output_path)

    print(f"Processing complete. Output saved as {output_path}")


if __name__ == "__main__":
    main()
