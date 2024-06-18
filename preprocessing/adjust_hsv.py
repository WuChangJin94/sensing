import cv2
import numpy as np

def nothing(x):
    pass

# Load the video
cap = cv2.VideoCapture('output_sample/sample2_flow_img.mp4')
if not cap.isOpened():
    print("Error opening video file")
    exit()

# Get video properties for resizing and display
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Create resizable windows for video display and trackbar adjustments
cv2.namedWindow('All combined video', cv2.WINDOW_NORMAL)
cv2.resizeWindow('All combined video', frame_width * 2, frame_height)  # Custom size

cv2.namedWindow('HSV Threshold Adjuster', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Hue Min', 'HSV Threshold Adjuster', 0, 179, nothing)
cv2.createTrackbar('Hue Max', 'HSV Threshold Adjuster', 179, 179, nothing)
cv2.createTrackbar('Sat Min', 'HSV Threshold Adjuster', 0, 255, nothing)
cv2.createTrackbar('Sat Max', 'HSV Threshold Adjuster', 255, 255, nothing)
cv2.createTrackbar('Val Min', 'HSV Threshold Adjuster', 0, 255, nothing)
cv2.createTrackbar('Val Max', 'HSV Threshold Adjuster', 255, 255, nothing)

paused = False  # Initial state of the video playback (not paused)
frame = None  # Variable to hold the current frame

while True:
    if not paused or frame is None:
        # Read a new frame
        ret, frame = cap.read()
        
        # If the frame is not received, it means we are at the end of the video
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Go back to the start
            continue
        
        source_image = frame[:frame.shape[0]//2, :, :]  # Upper half for source image
        gmflow_output = frame[frame.shape[0]//2:, :, :]  # Lower half for processing

    # Process HSV adjustment whether paused or not
    hue_min = cv2.getTrackbarPos('Hue Min', 'HSV Threshold Adjuster')
    hue_max = cv2.getTrackbarPos('Hue Max', 'HSV Threshold Adjuster')
    sat_min = cv2.getTrackbarPos('Sat Min', 'HSV Threshold Adjuster')
    sat_max = cv2.getTrackbarPos('Sat Max', 'HSV Threshold Adjuster')
    val_min = cv2.getTrackbarPos('Val Min', 'HSV Threshold Adjuster')
    val_max = cv2.getTrackbarPos('Val Max', 'HSV Threshold Adjuster')

    hsv = cv2.cvtColor(gmflow_output, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, (hue_min, sat_min, val_min), (hue_max, sat_max, val_max))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask_copy = mask.copy()

    # Find largest component for segmentation
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, 4, cv2.CV_32S)
    if num_labels > 1:
        largest_component = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        mask = (labels == largest_component).astype(np.uint8) * 255
        x, y, w, h = cv2.boundingRect(mask)

    # Apply mask to the source image
    masked_image = cv2.bitwise_and(source_image, source_image, mask=mask_copy)

    # Draw bounding box on source image
    source_bbox = source_image.copy()
    cv2.rectangle(source_bbox, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Combine all four parts into one frame for display
    top_row = np.hstack((source_image, gmflow_output))
    bottom_row = np.hstack((source_bbox, masked_image))
    combined_frame = np.vstack((top_row, bottom_row))

    cv2.imshow('All combined video', combined_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord(' '):
        paused = not paused  # Toggle the paused state

cap.release()
cv2.destroyAllWindows()
