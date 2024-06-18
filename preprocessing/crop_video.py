# !/usr/bin/env python3
import cv2
import numpy as np
import json
import argparse
import os

# Set up argument parsing
parser = argparse.ArgumentParser(description="Process a video to show various processing results in quadrants.")
parser.add_argument('-i', '--input', type=str, default="output_sample_0522/aerator_flow_img.mp4", help='Path to the input video file')
parser.add_argument('-o', '--output', type=str, default="cropped_output_0522/aerator_flow_img.mp4", help='Path to the output cropped video file')
parser.add_argument('-j', '--json', type=str, default="cropped_output_0522/aerator_flow_img.json", help='Path to the output cropped info file')
args = parser.parse_args()

# Initialize the list to store the points of rectangle
rect_pts = []  
cropping = False  # State to check if cropping is being done

os.makedirs(os.path.dirname(args.output), exist_ok=True)

# Open the video file
cap = cv2.VideoCapture(args.input)

# Get video properties
fps = int(cap.get(cv2.CAP_PROP_FPS))

# Check if video opened successfully
if not cap.isOpened():
    print("Error opening video stream or file")

def click_and_crop(event, x, y, flags, param):
    # Grab references to the global variables
    global rect_pts, cropping, frame

    # If the left mouse button was clicked, record the starting (x, y) coordinates
    if event == cv2.EVENT_LBUTTONDOWN:
        rect_pts = [(x, y)]
        cropping = True

    # Check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # Record the ending (x, y) coordinates and cropping state
        rect_pts.append((x, y))
        cropping = False

        # Draw a rectangle around the region of interest
        cv2.rectangle(frame, rect_pts[0], rect_pts[1], (0, 255, 0), 2)
        cv2.imshow("Video", frame)

def save_cropped_video_and_info(cap, rect_pts):
    # Calculate dimensions for the cropped area
    x, y = rect_pts[0][0], rect_pts[0][1]
    w, h = rect_pts[1][0] - x, rect_pts[1][1] - y

    # Prepare the crop info dictionary
    crop_info = {
        "x": x,
        "y": y,
        "width": w,
        "height": h
    }

    # Write crop information to JSON file
    with open(args.json, 'w') as json_file:
        json.dump(crop_info, json_file)
    
    # Notify user
    print(f"Crop info saved to {args.json}: {crop_info}")

    # Set up the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, fps, (w, h))

    # Read from the video capture
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Go to the beginning
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Crop and write the frame
        cropped_frame = frame[y:y+h, x:x+w]
        out.write(cropped_frame)

    # Release everything when job is finished
    out.release()
    print("Cropped video has been saved.")

# Set up the window and bind the function to window
cv2.namedWindow("Video")
cv2.setMouseCallback("Video", click_and_crop)

# Keep looping over the frames of the video
while True:
    ret, frame = cap.read()
    if not ret:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Reset the frame counter to loop the video
        continue

    # If a rectangle has been marked, draw it on the frame
    if len(rect_pts) == 2:
        cv2.rectangle(frame, rect_pts[0], rect_pts[1], (0, 255, 0), 2)

    # Display the image
    cv2.imshow("Video", frame)
    key = cv2.waitKey(1) & 0xFF

    # If the 'Enter' key is pressed and two points have been selected, process the crop
    if key == 13 and len(rect_pts) == 2:  # 13 is the Enter key
        save_cropped_video_and_info(cap, rect_pts)
        break

    # If the 'q' key is pressed, break from the loop
    if key == ord("q"):
        break

# Cleanup the camera and close any open windows
cap.release()
cv2.destroyAllWindows()
