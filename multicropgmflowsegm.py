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
parser.add_argument('-s', '--segmentation', type=str, default="segmentation_output_sample_0522/aeratorcompare10M_flow_img_segm_out.mp4", help='Path to the segmentation input video file')
parser.add_argument('-i', '--input', type=str, default="sample_files_0522/aeratorcompare10M_flow_img.mp4", help='Path to the input video file')
parser.add_argument('-o', '--output', type=str, default="detr_gmflow_final_output_0522/aeratorcompare10M_flow_img.mp4", help='Path to the output video file')
args = parser.parse_args()

# Create output directories
output_dir = os.path.splitext(args.output)[0]
os.makedirs(output_dir, exist_ok=True)

# Open the original and segmentation videos
cap_original = cv2.VideoCapture(args.input)
cap_segmentation = cv2.VideoCapture(args.segmentation)

# Open the cropped videos
cap_cropped_list = [cv2.VideoCapture(cropped_input) for cropped_input in args.cropped_input]

# Check if video opened successfully
if not cap_original.isOpened() or not cap_segmentation.isOpened() or not all(cap.isOpened() for cap in cap_cropped_list):
    print("Error opening video streams or files")
    if not cap_original.isOpened():
        print("Original video not opened")
    if not cap_segmentation.isOpened():
        print("Segmentation video not opened")
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
out = cv2.VideoWriter(args.output, fourcc, fps, (frame_width_orig * 2, frame_height_orig * 2))

# Load crop info from JSON
crop_info_list = []
for json_path in args.json:
    with open(json_path, 'r') as f:
        crop_info_list.append(json.load(f))

frame_id = 0
# Process each frame of the video
while True:
    ret_orig, frame_orig = cap_original.read()
    ret_seg, frame_seg = cap_segmentation.read()
    if not ret_orig or not ret_seg:
        break

    # Initialize white frame
    white_frame_orig = np.ones_like(frame_orig) * 255
    segmentation_mask = cv2.cvtColor(frame_seg, cv2.COLOR_BGR2GRAY)
    _, segmentation_mask = cv2.threshold(segmentation_mask, 127, 255, cv2.THRESH_BINARY)
    
    # Ensure segmentation_mask is CV_8U
    segmentation_mask = segmentation_mask.astype(np.uint8)

    # Resize segmentation_mask to match white_frame_orig dimensions
    segmentation_mask_resized = cv2.resize(segmentation_mask, (frame_width_orig, frame_height_orig))

    # Process each cropped video
    for i, cap_cropped in enumerate(cap_cropped_list):
        ret_crop, frame_crop = cap_cropped.read()
        if not ret_crop:
            continue

        # Split frame into top and bottom halves
        mid_point = frame_crop.shape[0] // 2
        crop_gmflow_output = frame_crop[mid_point:, :, :]

        # Masked image (apply mask to gmflow_output)
        crop_info = crop_info_list[i]

        # Resize crop_gmflow_output to match crop_info dimensions
        if (crop_gmflow_output.shape[0] != crop_info['height']) or (crop_gmflow_output.shape[1] != crop_info['width']):
            crop_gmflow_output = cv2.resize(crop_gmflow_output, (crop_info['width'], crop_info['height']))
        
        white_frame_orig[crop_info['y']:crop_info['y'] + crop_info['height'], crop_info['x']:crop_info['x'] + crop_info['width'], :] = crop_gmflow_output
        
    segmentation_applied = cv2.bitwise_and(white_frame_orig, white_frame_orig, mask=segmentation_mask_resized)
    
    # Combine all four parts into one frame
    source_image = frame_orig
    segmentation_mask_colored = cv2.cvtColor(segmentation_mask_resized, cv2.COLOR_GRAY2BGR)
    top_row = np.hstack((source_image, white_frame_orig))
    bottom_row = np.hstack((segmentation_mask_colored, segmentation_applied))
    combined_frame = np.vstack((top_row, bottom_row))

    # Write the frame to the output video
    out.write(combined_frame)

    # Save combined frame to disk
    frame_filename = os.path.join(output_dir, f"frame_{frame_id:04d}.jpg")
    cv2.imwrite(frame_filename, combined_frame)
    print(f"Processed frame {frame_id}")
    frame_id += 1

# Release resources and close files
cap_original.release()
for cap in cap_cropped_list:
    cap.release()
cap_segmentation.release()
out.release()
