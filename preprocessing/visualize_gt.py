import cv2
import json
import os
import numpy as np
import argparse

# Set up argument parsing
parser = argparse.ArgumentParser(description="Process a video to show various processing results in quadrants.")
parser.add_argument('-j', '--json', type=str, default="labels_output_0527/flow_1/annotations/flow_1.json", help='Path to the ground truth JSON file')
parser.add_argument('-i', '--images_dir', type=str, default="labels_output_0527/flow_1/images", help='Path to the directory containing the images')
parser.add_argument('-o', '--output', type=str, default="labels_output_0527/flow_1/flow_1_labels.mp4", help='Path to the output video file')
args = parser.parse_args()

# Load the JSON file
with open(args.json, 'r') as f:
    data = json.load(f)

# Get the list of images
image_files = sorted(os.listdir(args.images_dir))

# Get the first image to determine the frame size
first_image = cv2.imread(os.path.join(args.images_dir, image_files[0]))
frame_height, frame_width, _ = first_image.shape

# Define the codec and create a VideoWriter object
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.output, fourcc, 30, (frame_width, frame_height))

# Create resizable windows for video display and trackbar adjustments
cv2.namedWindow('Visualize annotations', cv2.WINDOW_NORMAL)
cv2.resizeWindow('Visualize annotations', frame_width, frame_height)  # Custom size

# Process each image and add the bounding boxes and segmentations
for image_file in image_files:
    image = cv2.imread(os.path.join(args.images_dir, image_file))

    # Get the annotations for this image
    image_id = int(image_file.split('_')[-1].split('.')[0])
    annotations = [annotation for annotation in data['annotations'] if annotation['image_id'] == image_id]

    # Draw the bounding boxes and segmentations
    for annotation in annotations:
        bbox = annotation['bbox']
        segmentation = annotation['segmentation']
        cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]), (0, 255, 0), 2)
        for seg in segmentation:
            seg = np.array(seg).reshape(-1, 2)
            cv2.polylines(image, [seg], isClosed=True, color=(255, 0, 0), thickness=2)

    # Display the image
    cv2.imshow('Visualize annotations', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Write the frame to the video
    out.write(image)

# Release the VideoWriter and destroy all windows
out.release()
# cv2.destroyAllWindows()
