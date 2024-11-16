import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from moviepy.editor import VideoFileClip
from tqdm import tqdm
import imutils

# Function to remove audio from a video
def remove_audio(input_video_path, output_video_path):
    video = VideoFileClip(input_video_path)
    video_no_audio = video.without_audio()
    video_no_audio.write_videofile(output_video_path, codec="libx264", audio=False)


# Function to cluster bounding boxes
def cluster_bounding_boxes(boxes, eps=50, min_samples=1):
    if not boxes:
        return []

    # Convert bounding boxes to (x_center, y_center) format for clustering
    centers = np.array([(x + w // 2, y + h // 2) for x, y, w, h in boxes])

    # Apply DBSCAN clustering
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(centers)

    # Group boxes by cluster labels
    clustered_boxes = []
    for cluster_id in set(clustering.labels_):
        if cluster_id == -1:  # Noise points
            continue

        cluster_indices = np.where(clustering.labels_ == cluster_id)[0]
        cluster_boxes = [boxes[i] for i in cluster_indices]

        # Merge all boxes in the cluster into one bounding box
        x_min = min([x for x, _, _, _ in cluster_boxes])
        y_min = min([y for _, y, _, _ in cluster_boxes])
        x_max = max([x + w for x, _, w, _ in cluster_boxes])
        y_max = max([y + h for _, y, _, h in cluster_boxes])

        clustered_boxes.append((x_min, y_min, x_max - x_min, y_max - y_min))

    return clustered_boxes


# Function to reduce color resolution
def quantize_colors(frame, k=24):
    """Reduces the number of colors in the image using k-means clustering."""
    # Reshape the image to a 2D array of pixels
    pixel_values = frame.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)

    # Apply k-means clustering
    _, labels, centers = cv2.kmeans(
        pixel_values, k, None,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        attempts=10, flags=cv2.KMEANS_RANDOM_CENTERS
    )

    # Convert centers to integers and reshape the image back
    centers = np.uint8(centers)
    quantized_frame = centers[labels.flatten()]
    quantized_frame = quantized_frame.reshape(frame.shape)

    return quantized_frame


# Function to detect license plates based on yellow color
def detect_license_plates(frame):
    # Reduce color space resolution
    quantized_frame = quantize_colors(frame, k=32)

    # Convert to HSL for color segmentation
    hsl_frame = cv2.cvtColor(quantized_frame, cv2.COLOR_BGR2HLS)

    # Define HSL range for yellow license plates
    lower_yellow = np.array([15, 100, 100])  # Adjusted for quantized yellow
    upper_yellow = np.array([35, 255, 255])

    # Create a mask for yellow regions
    yellow_mask = cv2.inRange(hsl_frame, lower_yellow, upper_yellow)

    # Apply morphological operations to clean the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_CLOSE, kernel)
    yellow_mask = cv2.morphologyEx(yellow_mask, cv2.MORPH_OPEN, kernel)

    return yellow_mask


# Updated function to blur license plates by replacing them with average color
def blur_license_plate(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Processing Frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Detect license plates
        yellow_mask = detect_license_plates(frame)

        # Apply mask to original frame and replace masked area with average color
        if np.any(yellow_mask):  # Check if mask has any detected regions
            # Compute average color in the masked region
            masked_area = cv2.bitwise_and(frame, frame, mask=yellow_mask)
            average_color = cv2.mean(masked_area, mask=yellow_mask)[:3]

            # Create an average color image to blend with the masked region
            average_color_image = np.full_like(frame, average_color, dtype=np.uint8)
            frame = cv2.bitwise_and(frame, frame, mask=cv2.bitwise_not(yellow_mask))  # Remove mask area
            frame += cv2.bitwise_and(average_color_image, average_color_image, mask=yellow_mask)  # Add averaged color

        out.write(frame)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()


def main():
    input_path = r"C:\Users\omerr\Downloads\prius_album\20241116_104132.mp4"
    no_audio_path = "car_video_no_audio.mp4"
    output_path = "car_video_blurred.mp4"

    # Step 1: Remove audio
    print("Removing audio from the video...")
    # remove_audio(input_path, no_audio_path)

    # Step 2: Blur license plates
    print("Blurring license plates in the video...")
    blur_license_plate(no_audio_path, output_path)

    print(f"Processing complete. Output saved as {output_path}")


if __name__ == "__main__":
    main()
