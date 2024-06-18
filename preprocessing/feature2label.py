# python3 feature2label.py -i output_sample/sample1_flow_img.mp4 -o labels_sample/sample1.mp4
import cv2
import numpy as np
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Process a video to show various processing results in quadrants.")
parser.add_argument('-i', '--input', type=str, default="output_sample_0522/aerator_flow_img.mp4", help='Path to the input video file')
parser.add_argument('-o', '--output', type=str, default="labels_output_0522/aerator_flow_img.mp4", help='Path to the output video file')
parser.add_argument('-h_min', '--hue_min', type=int, default=0, help='hue min value')
parser.add_argument('-s_min', '--sat_min', type=int, default=0, help='sat min value')
parser.add_argument('-v_min', '--val_min', type=int, default=0, help='val min value')
parser.add_argument('-h_max', '--hue_max', type=int, default=255, help='hue max value')
parser.add_argument('-s_max', '--sat_max', type=int, default=255, help='sat max value')
parser.add_argument('-v_max', '--val_max', type=int, default=255, help='val max value')
args = parser.parse_args()

# Open the video file
cap = cv2.VideoCapture(args.input)

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

# Get video properties
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Define the output video (double width and height to accommodate four segments)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width * 2, frame_height))

# Color range for segmentation (adjust as needed)
hue_min, sat_min, val_min = args.hue_min, args.sat_min, args.val_min  # Minimum HSV values
hue_max, sat_max, val_max = args.hue_max, args.sat_max, args.val_max  # Maximum HSV values

# Create resizable windows for video display and trackbar adjustments
cv2.namedWindow('Original frame', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Original frame', frame_width, frame_height)  # Custom size

# Process each frame of the video
while cap.isOpened():
    ret, frame = cap.read()
    cv2.imshow('Original frame', frame)
    if not ret:
        break

    # Split frame into top and bottom halves
    mid_point = frame_height // 2
    source_image = frame[:mid_point, :, :]
    gmflow_output = frame[mid_point:, :, :]

    # GMFlow processing to generate mask
    hsv = cv2.cvtColor(gmflow_output, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (hue_min, sat_min, val_min), (hue_max, sat_max, val_max))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Find largest component for segmentation
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    if num_labels > 1:
        largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        mask = (labels == largest_component).astype(np.uint8) * 255
        x, y, w, h = cv2.boundingRect(mask)

    # Masked image (apply mask to source image)
    masked_image = cv2.bitwise_and(source_image, source_image, mask=mask)

    # Draw bounding box on source image
    source_bbox = source_image.copy()
    cv2.rectangle(source_bbox, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Combine all four parts into one frame
    top_row = np.hstack((source_image, gmflow_output))
    bottom_row = np.hstack((source_bbox, masked_image))
    combined_frame = np.vstack((top_row, bottom_row))

    # cv2.imshow('All combined video', combined_frame)

    # Write the frame to the output video
    out.write(combined_frame)

# Release resources and close files
cap.release()
out.release()
# cv2.destroyAllWindows()
