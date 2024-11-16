import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from moviepy.editor import VideoFileClip
from tqdm import tqdm


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


# Function to apply flood-fill and watershed segmentation
def refine_license_plate_region(frame, mask, x, y, w, h):
    # Create a marker image for watershed
    markers = np.zeros_like(mask, dtype=np.int32)

    # Set the background region to 1
    cv2.rectangle(markers, (0, 0), (frame.shape[1], frame.shape[0]), 1, thickness=-1)

    # Set the license plate region to a unique value (e.g., 2)
    cv2.rectangle(markers, (x, y), (x + w, y + h), 2, thickness=-1)

    # Convert the frame to grayscale for watershed input
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply watershed segmentation
    cv2.watershed(frame, markers)

    # Create a mask for the license plate region after segmentation
    refined_mask = (markers == 2).astype(np.uint8) * 255
    return refined_mask


# Function to detect and blur license plates
def blur_license_plate(input_video_path, output_video_path):
    cap = cv2.VideoCapture(input_video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video_path, fourcc, int(cap.get(cv2.CAP_PROP_FPS)),
                          (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    # Define the color range for the license plate
    lower_color = np.array([20, 50, 50])  # Lower bound of yellow-orange in HSL
    upper_color = np.array([40, 255, 255])  # Upper bound of yellow-orange in HSL

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = tqdm(total=total_frames, desc="Processing Frames")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to HSL
        hsl_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)

        # Mask the license plate based on color
        mask = cv2.inRange(hsl_frame, lower_color, upper_color)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Extract bounding rectangles
        boxes = [cv2.boundingRect(contour) for contour in contours if cv2.contourArea(contour) > 50]

        # Cluster and merge bounding boxes
        clustered_boxes = cluster_bounding_boxes(boxes, eps=60, min_samples=1)

        for (x, y, w, h) in clustered_boxes:
            # Refine the mask using flood-fill and watershed
            refined_mask = refine_license_plate_region(frame, mask, x, y, w, h)

            # Blur the refined license plate region
            blur_frame = frame.copy()
            blur_frame[refined_mask == 255] = cv2.GaussianBlur(frame[refined_mask == 255], (51, 51), 30)
            frame[refined_mask == 255] = blur_frame[refined_mask == 255]

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
