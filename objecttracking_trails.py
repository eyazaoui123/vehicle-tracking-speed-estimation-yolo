#Plotting Tracks Over Time
import math

#Import All the Required Libraries
import cv2
from ultralytics import YOLO
from collections import defaultdict
import numpy as np
from collections import deque
import cv2
import time
from numpy import random

import argparse

# Command-line arguments setup
parser = argparse.ArgumentParser()
parser.add_argument('--hide_labels', action='store_true', help='Hide labels in the output')
parser.add_argument('--hide_conf', action='store_true', help='Hide confidence values in the output')
args = parser.parse_args()

print(dir(cv2))


palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)

object_counter = {}

object_counter1 = {}

speed_line_queue = {}
line = [(100, 500), (1050, 500)]

classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
                  "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
                  "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
                  "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
                  "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"]


def estimatespeed(Location1, Location2):
    d_pixel = math.sqrt(math.pow(Location2[0] - Location1[0], 2) + math.pow(Location2[1] - Location1[1], 2))
    # defining the pixels per meter
    ppm = 6
    d_meters = d_pixel/ppm
    time_constant = 15*3.6

    speed = d_meters * time_constant

    return int(speed)


def UI_box(box, img, color=None, label=None, line_thickness=None):
    """
    Draws a bounding box on the image.

    Parameters:
        box (tuple): (x, y, w, h) where (x, y) is the top-left corner, and (w, h) are width and height.
        img (numpy array): The image to draw on.
        color (tuple): The color of the bounding box (B, G, R).
        label (str): The label to display on the bounding box.
        line_thickness (int): Thickness of the bounding box lines.
    """
    # Default line thickness if not provided
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1

    # Default color if not provided
    color = color or [random.randint(0, 255) for _ in range(3)]

    # Convert (x, y, w, h) to (x1, y1, x2, y2)
    x, y, w, h = box
    x1, y1 = int(x-w/2), int(y-h/2)  # Top-left corner
    x2, y2 = int(x + w/2), int(y + h/2)  # Bottom-right corner

    # Draw the bounding box
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness=tl, lineType=cv2.LINE_AA)

    # Draw the label if provided
    if label:
        tf = max(tl - 1, 1)  # Font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]  # Get text size

        # Draw a background for the label
        img = draw_border(img, (x1, y1 - t_size[1] - 3), (x1 + t_size[0], y1 + 3), color, 1, 8, 2)

        # Draw the label text
        cv2.putText(img, label, (x1, y1 - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)

def compute_color_for_labels(label):
    """
    Simple function that adds fixed color depending on the class
    """
    if label == 0: #person
        color = (85,45,255)
    elif label == 1:#bicycle
        color = (0, 255, 0)
    elif label == 2: # Car
        color = (222,82,175)
    elif label == 3:  # Motobike
        color = (0, 204, 255)
    elif label == 5:  # Bus
        color = (0, 149, 255)
    elif label == 7:  # truck
        color = (255, 0, 0)
    else:
        color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)


def draw_border(img, pt1, pt2, color, thickness, r, d):
    x1, y1 = pt1
    x2, y2 = pt2
    # Top left
    cv2.line(img, (x1 + r, y1), (x1 + r + d, y1), color, thickness)
    cv2.line(img, (x1, y1 + r), (x1, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x1 + r, y1 + r), (r, r), 180, 0, 90, color, thickness)
    # Top right
    cv2.line(img, (x2 - r, y1), (x2 - r - d, y1), color, thickness)
    cv2.line(img, (x2, y1 + r), (x2, y1 + r + d), color, thickness)
    cv2.ellipse(img, (x2 - r, y1 + r), (r, r), 270, 0, 90, color, thickness)
    # Bottom left
    cv2.line(img, (x1 + r, y2), (x1 + r + d, y2), color, thickness)
    cv2.line(img, (x1, y2 - r), (x1, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x1 + r, y2 - r), (r, r), 90, 0, 90, color, thickness)
    # Bottom right
    cv2.line(img, (x2 - r, y2), (x2 - r - d, y2), color, thickness)
    cv2.line(img, (x2, y2 - r), (x2, y2 - r - d), color, thickness)
    cv2.ellipse(img, (x2 - r, y2 - r), (r, r), 0, 0, 90, color, thickness)

    cv2.rectangle(img, (x1 + r, y1), (x2 - r, y2), color, -1, cv2.LINE_AA)
    cv2.rectangle(img, (x1, y1 + r), (x2, y2 - r - d), color, -1, cv2.LINE_AA)

    cv2.circle(img, (x1 + r, y1 + r), 2, color, 12)
    cv2.circle(img, (x2 - r, y1 + r), 2, color, 12)
    cv2.circle(img, (x1 + r, y2 - r), 2, color, 12)
    cv2.circle(img, (x2 - r, y2 - r), 2, color, 12)

    return img


def get_direction(point1, point2):
    direction_str = ""

    # calculate y axis direction
    if point1[1] > point2[1]:
        direction_str += "South"
    elif point1[1] < point2[1]:
        direction_str += "North"
    else:
        direction_str += ""

    # calculate x axis direction
    if point1[0] > point2[0]:
        direction_str += "East"
    elif point1[0] < point2[0]:
        direction_str += "West"
    else:
        direction_str += ""

    return direction_str

def intersect(A,B,C,D):
    return ccw(A,C,D) != ccw(B,C,D) and ccw(A,B,C) != ccw(A,B,D)

def ccw(A,B,C):
    return (C[1]-A[1]) * (B[0]-A[0]) > (B[1]-A[1]) * (C[0]-A[0])


#Load the YOLO Model
# Charger le modèle YOLO pour la segmentation
model = YOLO("yolo11n-seg.pt")  # Remplacez par le chemin de votre modèle de segmentation
#Create a Video Capture Object
cap = cv2.VideoCapture("Resources/Videos/video7.mp4")
frame_width=int(cap.get(3))
frame_height=int(cap.get(4))
#Store the Track History
track_history = defaultdict(lambda: deque(maxlen=30))
import os

# Create the videos directory if it doesn't exist
#if not os.path.exists('.C:/Users/MEGA PC/Desktop/stage/yolo11_learning/8_MultiObjectTracking/videos'):
    #os.makedirs('C:/Users/MEGA PC/Desktop/stage/yolo11_learning/8_MultiObjectTracking/videos')

out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 10, (frame_width, frame_height))
object_speed=0
ptime = 0
ctime = 0
#Loop through the Video Frames
while True:
    ret, frame = cap.read()
    if ret:
        # Run YOLO tracking on the frame with segmentation
        results = model.track(source=frame, persist=True, save=False,
                              tracker="bytetrack.yaml")  # Use a tracker like ByteTrack

        # Visualize the results on the annotated frame
        annotated_frame = results[0].plot()  # This will plot bounding boxes and labels

        # Get the masks if they exist (for segmentation)
        if results[0].masks is not None:
            masks = results[0].masks.data.cpu().numpy()  # Extract masks
            for mask in masks:
                # Convert the mask to a binary image
                mask = (mask > 0.5).astype(np.uint8) * 255

                # Debug: Print shapes
                print("Frame shape:", frame.shape)
                print("Mask shape:", mask.shape)

                # Resize the mask to match the frame dimensions
                mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))  # Resize to (width, height)

                # Convert the mask to a 3-channel image (BGR)
                mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

                # Debug: Print shapes after conversion
                print("Mask BGR shape:", mask_bgr.shape)

                # Overlay the mask on the frame
                frame = cv2.addWeighted(frame, 1, mask_bgr, 0.5, 0)
        # Draw the line for counting objects
        cv2.line(frame, line[0], line[1], (46, 162, 112), 3)
        height, width, _ = frame.shape

        # Check if there are tracked objects
        if results[0].boxes.id is not None:
            # Get the bounding box coordinates, track IDs, labels, and confidences
            boxes = results[0].boxes.xywh.cpu()
            labels = results[0].boxes.cls.int()
            confs = results[0].boxes.conf.float()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # For each detected object
            for box, track_id, label, conf in zip(boxes, track_ids, labels, confs):
                classname = classNames[label]
                x, y, w, h = box

                # Update track history
                track = track_history[track_id]
                track.appendleft((float(x), float(y)))  # x, y center point
                if len(track) > 30:
                    track.pop()

                # Calculate direction and speed if enough points are available
                if len(track_history[track_id]) >= 2:
                    direction = get_direction(track_history[track_id][0], track_history[track_id][1])

                    # Check if the object crosses the line
                    if intersect(track[0], track[1], line[0], line[1]):
                        cv2.line(frame, line[0], line[1], (255, 255, 255), 3)
                        object_speed = estimatespeed(track[1], track[0])

                        # Initialize speed_line_queue for the track_id if it doesn't exist
                        if track_id not in speed_line_queue:
                            speed_line_queue[track_id] = deque(maxlen=30)
                        speed_line_queue[track_id].appendleft(object_speed)

                        # Update counters based on direction
                        if "South" in direction:
                            object_counter[classname] = object_counter.get(classname, 0) + 1
                        if "North" in direction:
                            object_counter1[classname] = object_counter1.get(classname, 0) + 1

                    # Assign color based on label
                    color = compute_color_for_labels(label)

                    # Calculate average speed
                    speed_info = speed_line_queue.get(track_id, [])
                    average_speed = sum(speed_info) // len(speed_info) if speed_info else 0

                    # Construct the label
                    if speed_info:  # Include speed if available
                        label = None if args.hide_labels else (
                            f'id:{track_id} {classname} {conf:.2f}' if args.hide_conf
                            else f'id:{track_id} {classname} {conf:.2f} {direction} {average_speed} km/s'
                        )
                    else:  # Exclude speed if not available
                        label = None if args.hide_labels else (
                            f'id:{track_id} {classname} {conf:.2f}' if args.hide_conf
                            else f'id:{track_id} {classname} {conf:.2f} {direction}'
                        )

                    # Draw the bounding box and label
                    UI_box(box, frame, label=label, color=color, line_thickness=2)

                # Draw the tracking lines
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=(230, 0, 0), thickness=5)

            # Display counters in the top right corner
            for idx, (key, value) in enumerate(object_counter1.items()):
                cnt_str = str(key) + ":" + str(value)
                cv2.line(frame, (width - 500, 25), (width, 25), [85, 45, 255], 40)
                cv2.putText(frame, f'Number of Vehicles Entering', (width - 500, 35), 0, 1, [225, 255, 255],
                            thickness=2, lineType=cv2.LINE_AA)
                cv2.line(frame, (width - 150, 65 + (idx * 40)), (width, 65 + (idx * 40)), [85, 45, 255], 30)
                cv2.putText(frame, cnt_str, (width - 150, 75 + (idx * 40)), 0, 1, [255, 255, 255], thickness=2,
                            lineType=cv2.LINE_AA)

            for idx, (key, value) in enumerate(object_counter.items()):
                cnt_str1 = str(key) + ":" + str(value)
                cv2.line(frame, (20, 25), (500, 25), [85, 45, 255], 40)
                cv2.putText(frame, f'Numbers of Vehicles Leaving', (11, 35), 0, 1, [225, 255, 255], thickness=2,
                            lineType=cv2.LINE_AA)
                cv2.line(frame, (20, 65 + (idx * 40)), (127, 65 + (idx * 40)), [85, 45, 255], 30)
                cv2.putText(frame, cnt_str1, (11, 75 + (idx * 40)), 0, 1, [225, 255, 255], thickness=2,
                            lineType=cv2.LINE_AA)

            # Print counters to the console
            print(f"Vehicles Entering Counter: {object_counter1}")
            print(f"Vehicles Leaving Counter: {object_counter}")

            # Save the frame to the output video
            out.write(frame)

            # Calculate and display FPS
            ctime = time.time()
            fps = 1 / (ctime - ptime)
            ptime = ctime
            cv2.putText(annotated_frame, "FPS" + ":" + str(int(fps)), (10, 150), cv2.FONT_HERSHEY_PLAIN, 3,
                        (255, 0, 255), 3)

            # Show the frame
            cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('w'):
                break
    else:
        break
out.release()
cv2.destroyAllWindows()

