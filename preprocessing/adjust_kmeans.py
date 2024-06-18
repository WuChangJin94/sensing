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

cv2.namedWindow('K-means Adjuster', cv2.WINDOW_NORMAL)
cv2.createTrackbar('Threshold', 'K-means Adjuster', 127, 255, nothing)
cv2.createTrackbar('num_components', 'K-means Adjuster', 1, 5, nothing)
cv2.createTrackbar('break_connections', 'K-means Adjuster', 0, frame_width, nothing)

paused = False  # Initial state of the video playback (not paused)
frame = None  # Variable to hold the current frame

def apply_kmeans(image, K):
    Z = image.reshape((-1, 3))
    Z = np.float32(Z)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    _, labels, centers = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    res = centers[labels.flatten()]
    return res.reshape((image.shape))

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

    threshold_value = cv2.getTrackbarPos('Threshold', 'K-means Adjuster')

    # Apply K-means clustering to the source image
    kmeans_mask = apply_kmeans(source_image, 2)
    kmeans_mask = cv2.cvtColor(kmeans_mask, cv2.COLOR_BGR2GRAY)
    _, kmeans_mask = cv2.threshold(kmeans_mask, threshold_value, 255, cv2.THRESH_BINARY)

    # Find largest component to break connections
    break_col = cv2.getTrackbarPos('break_connections', 'K-means Adjuster')
    if break_col > 0:
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(kmeans_mask, 4, cv2.CV_32S)
        if num_labels > 1:
            sorted_idx = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1  # Sort component areas in descending order
            largest_idx = sorted_idx[0]
            largest_mask = np.zeros_like(kmeans_mask)
            largest_mask[labels == largest_idx] = 255
            x, y, w, h = cv2.boundingRect(largest_mask)
            break_col_within_bbox = x + break_col
            if break_col_within_bbox < x + w:
                largest_mask[:, break_col_within_bbox] = 0
            kmeans_mask = largest_mask

    # Find n largest components for segmentation
    num_components = cv2.getTrackbarPos('num_components', 'K-means Adjuster')
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(kmeans_mask, 4, cv2.CV_32S)
    if num_labels > 1:
        sorted_idx = np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1  # Sort component areas in descending order
        keep_idx = sorted_idx[:num_components]  # Keep the top `num_components`
        final_mask = np.zeros_like(kmeans_mask)
        for idx in keep_idx:
            final_mask[labels == idx] = 255
        mask = final_mask
        x, y, w, h = cv2.boundingRect(mask)
    elif num_labels == 1:
        mask = kmeans_mask
        x, y, w, h = cv2.boundingRect(mask)
    else:
        raise ValueError("No components found in mask")

    # Apply mask to the source image
    masked_image = cv2.bitwise_and(source_image, source_image, mask=mask)

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
