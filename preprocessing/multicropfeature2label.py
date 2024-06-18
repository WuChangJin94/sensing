import cv2
import numpy as np
import argparse
import json
import os

# Set up argument parsing
parser = argparse.ArgumentParser(description="Process a video to show various processing results in quadrants.")
parser.add_argument('-c', '--cropped_input', type=str, nargs='+', default=[
    "cropped_output_sample_0522/aeratorcompare10M_flow_img_1.mp4",
    "cropped_output_sample_0522/aeratorcompare10M_flow_img_2.mp4",
    "cropped_output_sample_0522/aeratorcompare10M_flow_img_3.mp4"], help='Paths to the cropped input video files')
parser.add_argument('-j', '--json', type=str, nargs='+', default=[
    "cropped_output_0522/aeratorcompare10M_flow_img_1.json",
    "cropped_output_0522/aeratorcompare10M_flow_img_2.json",
    "cropped_output_0522/aeratorcompare10M_flow_img_3.json"], help='Paths to the output cropped info files')
parser.add_argument('-i', '--input', type=str, default="output_sample_0522/aeratorcompare10M_flow_img.mp4", help='Path to the input video file')
parser.add_argument('-o', '--output', type=str, default="labels_output_0522/aeratorcompare10M_flow_img.mp4", help='Path to the output video file')
parser.add_argument('--use_hsv', action='store_true', help='Enable HSV segmentation')
parser.add_argument('-h_min', '--hue_min', type=int, nargs='+', default=[0, 0, 0], help='Hue min values')
parser.add_argument('-s_min', '--sat_min', type=int, nargs='+', default=[0, 0, 0], help='Sat min values')
parser.add_argument('-v_min', '--val_min', type=int, nargs='+', default=[0, 0, 0], help='Val min values')
parser.add_argument('-h_max', '--hue_max', type=int, nargs='+', default=[179, 179, 179], help='Hue max values')
parser.add_argument('-s_max', '--sat_max', type=int, nargs='+', default=[255, 255, 255], help='Sat max values')
parser.add_argument('-v_max', '--val_max', type=int, nargs='+', default=[255, 255, 255], help='Val max values')
parser.add_argument('--use_kmeans', action='store_true', help='Enable K-means segmentation')
parser.add_argument('-kmeans_t', '--kmeans_threshold', type=int, nargs='+', default=[127, 127, 127], help='K-means threshold value')
parser.add_argument('-n', '--num_components', type=int, nargs='+', default=[1, 1, 1], help='Number of largest components to keep')
parser.add_argument('-b', '--break_connections', type=int, nargs='+', default=[0, 0, 0], help='Break connections between components')
args = parser.parse_args()

# Create output directories
output_dir = os.path.splitext(args.output)[0]
images_dir = os.path.join(output_dir, 'images')
annotations_dir = os.path.join(output_dir, 'annotations')
os.makedirs(images_dir, exist_ok=True)
os.makedirs(annotations_dir, exist_ok=True)

# Initialize COCO annotation structure
coco_output = {
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": 1,
            "name": "water_splash"
        }
    ]
}
annotation_id = 1
image_id = 1

# Load crop info from JSON
crop_info_list = []
for json_path in args.json:
    with open(json_path, 'r') as f:
        crop_info_list.append(json.load(f))

# Open the original video
cap_original = cv2.VideoCapture(args.input)

# Open the cropped videos
cap_cropped_list = [cv2.VideoCapture(cropped_input) for cropped_input in args.cropped_input]

# Check if video opened successfully
if not cap_original.isOpened() or not all(cap.isOpened() for cap in cap_cropped_list):
    print("Error opening video streams or files")
    if not cap_original.isOpened():
        print("Original video not opened")
    for i, cap in enumerate(cap_cropped_list):
        if not cap.isOpened():
            print(f"Cropped video {i+1} not opened")
    exit(1)

# Get video properties from the original video
frame_width_orig = int(cap_original.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height_orig = int(cap_original.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap_original.get(cv2.CAP_PROP_FPS))

# Setup output video
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width_orig * 2, frame_height_orig))

# Color range for segmentation (adjust as needed)
hue_min_list, sat_min_list, val_min_list = args.hue_min, args.sat_min, args.val_min  # Minimum HSV values
hue_max_list, sat_max_list, val_max_list = args.hue_max, args.sat_max, args.val_max  # Maximum HSV values

# Create resizable windows for video display and trackbar adjustments
cv2.namedWindow('All combined video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('All combined video', frame_width_orig * 2, frame_height_orig)  # Custom size

# Function to apply K-means clustering segmentation
def apply_kmeans(image, K=2):
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

# Function to break connections between components
def break_connections(mask, break_col):
    # Find largest component to break connections
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    sorted_idx = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1  # Sort component areas in descending order
    keep_idx = sorted_idx[:1]
    mask = np.zeros_like(mask)
    for idx in keep_idx:
        mask[labels == idx] = 255
    x, y, w, h = cv2.boundingRect(mask)
    mask[:, x + break_col] = 0
    return mask

# Process each frame of the video
frame_id = 0
while True:
    ret_orig, frame_orig = cap_original.read()
    if not ret_orig:
        break

    frame_id += 1
    # Initialize black and white frames outside the loop
    black_frame_orig = np.zeros_like(frame_orig[frame_height_orig // 2:, :, :])
    white_frame_orig = np.ones_like(frame_orig[:frame_height_orig // 2, :, :]) * 255

    bbox_list = []
    segmentation_img = np.zeros_like(frame_orig[:frame_height_orig // 2, :, 0])

    # Process each cropped video
    for i, cap_cropped in enumerate(cap_cropped_list):
        ret_crop, frame_crop = cap_cropped.read()
        if not ret_crop:
            continue

        # Split frame into top and bottom halves
        mid_point = frame_crop.shape[0] // 2
        crop_source_image = frame_crop[:mid_point, :, :]
        crop_gmflow_output = frame_crop[mid_point:, :, :]

        hsv_mask = None
        kmeans_mask = None

        if args.use_hsv:
            # HSV processing to generate mask
            hsv = cv2.cvtColor(crop_gmflow_output, cv2.COLOR_BGR2HSV)
            hsv_mask = cv2.inRange(hsv, (hue_min_list[i], sat_min_list[i], val_min_list[i]), (hue_max_list[i], sat_max_list[i], val_max_list[i]))
            kernel = np.ones((5, 5), np.uint8)
            hsv_mask = cv2.morphologyEx(hsv_mask, cv2.MORPH_OPEN, kernel)
            if args.break_connections[i] > 0:
                hsv_mask = break_connections(hsv_mask, args.break_connections[i])
        
        if args.use_kmeans:
            # Apply K-means clustering to the source image
            kmeans_mask = apply_kmeans(crop_source_image, K=2)
            kmeans_mask = cv2.cvtColor(kmeans_mask, cv2.COLOR_BGR2GRAY)
            _, kmeans_mask = cv2.threshold(kmeans_mask, args.kmeans_threshold[i], 255, cv2.THRESH_BINARY)
            if args.break_connections[i] > 0:
                kmeans_mask = break_connections(kmeans_mask, args.break_connections[i])

        if args.use_hsv and args.use_kmeans:
            mask = cv2.bitwise_and(hsv_mask, kmeans_mask)
        elif args.use_hsv:
            mask = hsv_mask
        elif args.use_kmeans:
            mask = kmeans_mask

        # Find n largest components for segmentation
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
        if num_labels > 1:
            sorted_idx = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1  # Sort component areas in descending order
            keep_idx = sorted_idx[:args.num_components[i]]  # Keep the top `num_components`
            final_mask = np.zeros_like(mask)
            for idx in keep_idx:
                final_mask[labels == idx] = 255
            mask = final_mask
            x, y, w, h = cv2.boundingRect(mask)
            bbox_list.append((x, y, w, h))
        elif num_labels == 1:
            x, y, w, h = cv2.boundingRect(mask)
            bbox_list.append((x, y, w, h))
        else:
            raise ValueError("No components found in mask")
        

        # Masked image (apply mask to source image)
        crop_masked_image = cv2.bitwise_and(crop_source_image, crop_source_image, mask=mask)

        crop_info = crop_info_list[i]
        if crop_info['width'] != crop_masked_image.shape[1] or crop_info['height'] != crop_masked_image.shape[0]:
            crop_masked_image = cv2.resize(crop_masked_image, (crop_info['width'], crop_info['height']))
            crop_gmflow_output = cv2.resize(crop_gmflow_output, (crop_info['width'], crop_info['height']))
            # Resize the mask to fit the original frame dimensions
            mask = cv2.resize(mask, (crop_info['width'], crop_info['height']))
        black_frame_orig[crop_info['y']:crop_info['y'] + crop_info['height'], crop_info['x']:crop_info['x'] + crop_info['width'], :] = crop_masked_image
        segmentation_img[crop_info['y']:crop_info['y'] + crop_info['height'], crop_info['x']:crop_info['x'] + crop_info['width']] = mask
        if args.use_hsv:
            white_frame_orig[crop_info['y']:crop_info['y'] + crop_info['height'], crop_info['x']:crop_info['x'] + crop_info['width'], :] = crop_gmflow_output

    # Draw bounding boxes on source image
    source_image = frame_orig[:frame_height_orig // 2, :, :]
    gmflow_output = frame_orig[frame_height_orig // 2:, :, :]
    source_bbox = source_image.copy()
    for crop_info, bbox in zip(crop_info_list, bbox_list):
        x, y = crop_info['x'] + bbox[0], crop_info['y'] + bbox[1]
        cv2.rectangle(source_bbox, (x, y), (x + bbox[2], y + bbox[3]), (0, 255, 0), 2)
        
        # Save cropped image
        image_filename = f"{os.path.basename(output_dir)}_{frame_id}.jpg"
        cv2.imwrite(os.path.join(images_dir, image_filename), source_image)

        # Create annotation
        contours, _ = cv2.findContours(segmentation_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        segmentation = []
        for contour in contours:
            if contour.size >= 6:  # Ensure there are at least 3 points (needed to form a polygon)
                # Adjust coordinates by adding crop_info['x'] and crop_info['y']
                adjusted_contour = [(int(point[0][0]), int(point[0][1])) for point in contour]
                segmentation.append(np.array(adjusted_contour).flatten().tolist())
        
        # Create annotation
        annotation = {
            "id": annotation_id,
            "image_id": image_id,
            "category_id": 1,
            "bbox": [x, y, bbox[2], bbox[3]],
            "segmentation": segmentation,
            "area": bbox[2] * bbox[3],
            "iscrowd": 0
        }
        coco_output["annotations"].append(annotation)
        annotation_id += 1
    
    # Add image entry
    coco_output["images"].append({
        "id": image_id,
        "file_name": image_filename,
        "width": frame_width_orig,
        "height": frame_height_orig // 2
    })
    image_id += 1

    # Combine all four parts into one frame
    top_row = np.hstack((source_image, white_frame_orig))
    bottom_row = np.hstack((source_bbox, black_frame_orig))
    combined_frame = np.vstack((top_row, bottom_row))

    cv2.imshow('All combined video', combined_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Write the frame to the output video
    out.write(combined_frame)

# Save COCO annotations
with open(os.path.join(annotations_dir, f"{os.path.basename(output_dir)}.json"), 'w') as f:
    json.dump(coco_output, f, indent=4)

# Release resources and close files
cap_original.release()
for cap in cap_cropped_list:
    cap.release()
out.release()
cv2.destroyAllWindows()
