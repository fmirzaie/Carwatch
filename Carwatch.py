
"""
The Carwatch project centers on detecting vehicles in both video recordings and live camera feeds using computer vision techniques, 
with a focus on motion detection. Leveraging OpenCV for sophisticated video processing and object detection, 
it is suited for applications such as traffic flow monitoring, enhancing security, and analyzing transportation patterns.

## Features and Key Concepts:
- Preprocessing: Converts frames to grayscale and applies Gaussian blur to simplify and smooth images for better detection.
- Motion Detection: Identifies changes between frames using frame differences to highlight motion.
- Contour Enhancement: Uses dilation to improve contour visibility and tracking.
- Bounding Box Merging: Combines overlapping boxes to avoid duplicate vehicle counts and ensures accurate tracking.
- Object Tracking: Measures distances between bounding boxes to track vehicles across frames.
- Region of Interest (ROI): Focuses processing on specific frame areas to improve accuracy and reduce computational load. 

"""

# imports
import cv2
import numpy as np


# Setup video source and output paths
input_video = 'video.mp4'  # Path to the input video file or camera feed
output_video = 'processed_video.mp4'  # Path for saving the processed video output


def preprocess_frame(frame):
    """
    Converts the frame to grayscale and applies Gaussian blur.

    Parameters:
    - frame: The input video frame in BGR format.

    Returns:
    - Blurred grayscale frame.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)
    return blurred_frame


def process_frame_difference(current_frame, previous_frame, threshold=25):
    """
    Computes the absolute difference between the current and previous frames and applies thresholding.

    Parameters:
    - current_frame: The current video frame.
    - previous_frame: The previous video frame.
    - threshold: The threshold for binary thresholding.

    Returns:
    - Binary frame highlighting differences.
    """
    diff_frame = cv2.absdiff(current_frame, previous_frame)
    _, thresh_frame = cv2.threshold(diff_frame, threshold, 255, cv2.THRESH_BINARY)
    return thresh_frame


def dilate_contours(thresh_frame, kernel_size=10, iterations=2):
    """
    Applies dilation to the binary thresholded frame to enhance contours.

    Parameters:
    - thresh_frame: The binary thresholded frame.
    - kernel_size: Size of the dilation kernel.
    - iterations: Number of dilation iterations.

    Returns:
    - Dilated frame with enhanced contours.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    dilated_frame = cv2.dilate(thresh_frame, kernel, iterations=iterations)
    return dilated_frame


def merge_bounding_boxes(boxes, threshold=20):
    """
    Merges overlapping or close bounding boxes into single boxes.

    Parameters:
    - boxes: List of bounding boxes to merge.
    - threshold: Distance threshold to determine overlap or closeness.

    Returns:
    - List of merged bounding boxes.
    """
    def overlap_or_close(box1, box2):
        x1, y1, w1, h1 = box1
        x2, y2, w2, h2 = box2
        return not (x1 + w1 < x2 - threshold or x2 + w2 < x1 - threshold or
                    y1 + h1 < y2 - threshold or y2 + h2 < y1 - threshold)

    def enclosing_box(group):
        x_min = y_min = float('inf')
        x_max = y_max = float('-inf')
        for box in group:
            x, y, w, h = box
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)
        return [x_min, y_min, x_max - x_min, y_max - y_min]

    groups = []
    visited = [False] * len(boxes)
    for i, box1 in enumerate(boxes):
        if visited[i]:
            continue
        group = [box1]
        stack = [i]
        while stack:
            idx = stack.pop()
            visited[idx] = True
            for j, box2 in enumerate(boxes):
                if not visited[j] and overlap_or_close(box1, box2):
                    group.append(box2)
                    stack.append(j)
        groups.append(group)

    merged_boxes = [enclosing_box(group) for group in groups]
    return merged_boxes


def calculate_distance(box1, box2):
    """
    Calculates the Euclidean distance between the centers of two bounding boxes.

    Parameters:
    - box1: First bounding box [x, y, w, h].
    - box2: Second bounding box [x, y, w, h].

    Returns:
    - Euclidean distance between the centers of the two bounding boxes.
    """
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    center1 = (x1 + w1 / 2, y1 + h1 / 2)
    center2 = (x2 + w2 / 2, y2 + h2 / 2)
    distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
    return distance


# Initialize video capture (change input_video to 0 for camera input)
cap = cv2.VideoCapture(input_video)

# Get video properties for saving output
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
fps = int(cap.get(cv2.CAP_PROP_FPS))
frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)

frame_no = 0
previous_frame = None
previous_boxes = []
detected_ids = set()
car_count = 0
distance_threshold = 50  # Threshold for matching boxes
playback_speed = 100  # Delay for frame playback in milliseconds
size_threshold = 2000  # Threshold for bounding box size to count as a car
processing_timestep = 5  # Interval at which frame processing is performed (every 5th frame)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_no += 1
    if frame_no % processing_timestep == 0:
        height, width, _ = frame.shape
        roi = frame[500: 720, 100: 800]
        current_frame = preprocess_frame(roi)
        if previous_frame is None:
            previous_frame = current_frame
            continue
        diff_frame = process_frame_difference(current_frame, previous_frame)

        # Dilate the contours in the binary mask
        dilated_frame = dilate_contours(diff_frame)

        contours, _ = cv2.findContours(dilated_frame, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:  # Minimum contour area to consider
                x, y, w, h = cv2.boundingRect(cnt)
                detections.append([x, y, w, h])

        merged_detections = merge_bounding_boxes(detections)

        new_boxes = []
        for current_box in merged_detections:
            matched = False
            for prev_box in previous_boxes:
                if calculate_distance(current_box, prev_box) < distance_threshold:
                    matched = True
                    break
            if not matched:
                new_boxes.append(current_box)


        previous_frame = current_frame
        previous_boxes = merged_detections

        key = cv2.waitKey(playback_speed)
        if key == 27:  # Escape key
            break

        # Detect cars
        for box in new_boxes:
            x, y, w, h = box
            area = w * h
            if area > size_threshold:  # Check if bounding box size is above threshold
                car_count += 1
                detected_ids.add(car_count)
                center_x = x + w // 2
                center_y = y + h // 2
                # Draw a green circle on the detected car
                cv2.circle(roi, (center_x, center_y), 20, (0, 255, 0), -1)
                # cv2.putText(roi, str(car_count), (center_x, center_y), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

        # Display the frame with number of detected cars
        cv2.putText(frame, f"#Vehicles: {car_count}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)
        cv2.imshow("Frame", frame)
        # cv2.imshow("dilated_frame", dilated_frame)

    else:
        # Display the number of detected cars on all frames
        cv2.putText(frame, f"#Vehicles: {car_count}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 255), 2)

    # Write the frame to the output video file
    out.write(frame)

cap.release()
out.release()

# Keep windows open until a key is pressed
while True:
    key = cv2.waitKey(0)
    if key == 27:  # Escape key
        break

cv2.destroyAllWindows()