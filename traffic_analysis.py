

!nvidia-smi


!pip install -q supervision
!pip install opencv-python-headless


!pip install -q supervision "ultralytics<=8.3.40"


# prevent ultralytics from tracking your activity
!yolo settings sync=False


import cv2


import numpy as np
import supervision as sv


from tqdm import tqdm
from ultralytics import YOLO
from supervision.assets import VideoAssets, download_assets
from collections import defaultdict, deque
import pandas as pd # Import pandas
import os # Import os for file existence check
import openpyxl # Import openpyxl
import datetime # Import datetime for timestamp formatting


SOURCE_VIDEO_PATH = '/content/drive/MyDrive/Acasia.mp4'
TARGET_VIDEO_PATH = '/content/drive/MyDrive/Acasiao_result_speed.mp4'
CONFIDENCE_THRESHOLD = 0.3
IOU_THRESHOLD = 0.5
MODEL_NAME = "yolov8x.pt"
MODEL_RESOLUTION = 1280


SOURCE = np.array([
    [1100, 287],
    [98, 403],
    [339, 709],
    [1000, 600]
])


TARGET_WIDTH = 15
TARGET_HEIGHT = 150


TARGET = np.array([
    [0, 0],
    [TARGET_WIDTH - 1, 0],
    [TARGET_WIDTH - 1, TARGET_HEIGHT - 1],
    [0, TARGET_HEIGHT - 1],
])


# Define ViewTransformer class
class ViewTransformer:


    def __init__(self, source: np.ndarray, target: np.ndarray) -> None:
        source = source.astype(np.float32)
        target = target.astype(np.float32)
        self.m = cv2.getPerspectiveTransform(source, target)


    def transform_points(self, points: np.ndarray) -> np.ndarray:
        if points.size == 0:
            return points


        reshaped_points = points.reshape(-1, 1, 2).astype(np.float32)
        transformed_points = cv2.perspectiveTransform(reshaped_points, self.m)
        return transformed_points.reshape(-1, 2)


# Instantiate ViewTransformer
view_transformer = ViewTransformer(source=SOURCE, target=TARGET)


model = YOLO(MODEL_NAME)


video_info = sv.VideoInfo.from_video_path(video_path=SOURCE_VIDEO_PATH)
frame_generator = sv.get_video_frames_generator(source_path=SOURCE_VIDEO_PATH)


# tracer initiation
byte_track = sv.ByteTrack(
    frame_rate=video_info.fps
)


# annotators configuration
thickness = sv.calculate_optimal_line_thickness( # Reverted to optimal
    resolution_wh=video_info.resolution_wh
)
text_scale = sv.calculate_optimal_text_scale( # Reverted to optimal
    resolution_wh=video_info.resolution_wh
)
bounding_box_annotator = sv.BoxAnnotator(
    thickness=thickness
)
label_annotator = sv.LabelAnnotator(
    text_scale=text_scale,
    text_thickness=thickness,
    text_position=sv.Position.BOTTOM_CENTER
)
trace_annotator = sv.TraceAnnotator(
    thickness=thickness,
    trace_length=video_info.fps * 2,
    position=sv.Position.BOTTOM_CENTER
)


polygon_zone = sv.PolygonZone(
    polygon=SOURCE
)


coordinates = defaultdict(lambda: deque(maxlen=video_info.fps))
# Dictionary to store historical positions for each tracked object
historical_positions = defaultdict(lambda: deque(maxlen=video_info.fps))
# Dictionary to store historical speeds for inferring evasive actions
historical_speeds = defaultdict(lambda: deque(maxlen=video_info.fps))


# Define movement threshold
movement_threshold = 1


# Function to determine direction based on historical positions
def determine_direction(history):
    if len(history) < 2:
        return "unknown"


    # Get the last two points in the history
    point1 = history[-2]
    point2 = history[-1]


    # Calculate the change in x and y coordinates in the transformed perspective
    dx = point2[0] - point1[0]
    dy = point2[1] - point1[1]


    # Determine direction based on transformed coordinates
    if abs(dx) < movement_threshold and abs(dy) < movement_threshold:
        return "stopped"
    # In the bird's-eye view (transformed perspective), straight movement is primarily along the y-axis
    elif abs(dy) > abs(dx) * 2: # Mostly vertical movement in transformed perspective
         return "straight"
    # Horizontal movement in the transformed perspective indicates turns
    elif abs(dx) > abs(dy) * 2: # Mostly horizontal movement in transformed perspective
        return "left turn" if dx < 0 else "right turn"
    else:
        # Diagonal movement, could be a turn or curve
        return "turning/curving" # More nuanced direction if needed


# Function to infer evasive action based on speed changes
def infer_evasive_action(speed_history):
    if len(speed_history) < 2:
        return "none"


    # Get the last two speed values
    speed1 = speed_history[-2]
    speed2 = speed_history[-1]


    speed_change_threshold = 5 # km/h threshold for significant speed change


    if speed2 < speed1 - speed_change_threshold:
        return "brake"
    elif speed2 > speed1 + speed_change_threshold:
        return "accelerate"
    elif speed2 < speed_change_threshold and speed1 < speed_change_threshold and len(speed_history) >= video_info.fps/2: # Check if vehicle has been slow for a duration
         return "stop" # Consider stopping if speed is consistently low
    # Add logic for 'swerve' if lateral movement data (dx in transformed points) is significant
    # This would require more complex analysis of lateral acceleration or sudden changes in direction
    # For simplicity, we are not implementing swerve detection based on speed history alone.
    else:
        return "none"


# Define the Excel file path
excel_file_path = 'traffic_conflicts.xlsx'


# Delete the existing file if it exists to ensure a clean start
if os.path.exists(excel_file_path):
    os.remove(excel_file_path)
    print(f"Deleted existing file: {excel_file_path}")


# open target video
with sv.VideoSink(TARGET_VIDEO_PATH, video_info) as sink:


    # loop over source video frame
    for frame_idx, frame in enumerate(tqdm(frame_generator, total=video_info.total_frames)):


        # Calculate timestamp for the current frame in seconds
        timestamp_seconds = frame_idx / video_info.fps
        # Format timestamp as HH:MM:SS
        timestamp_formatted = str(datetime.timedelta(seconds=timestamp_seconds))


        result = model(frame, imgsz=MODEL_RESOLUTION, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)


        # filter out detections by class and confidence
        detections = detections[detections.confidence > CONFIDENCE_THRESHOLD]
        detections = detections[detections.class_id != 0]


        # filter out detections outside the zone
        detections = detections[polygon_zone.trigger(detections)]


        # refine detections using non-max suppression
        detections = detections.with_nms(IOU_THRESHOLD)


        # pass detection through the tracker
        detections = byte_track.update_with_detections(detections=detections)


        points = detections.get_anchors_coordinates(
            anchor=sv.Position.BOTTOM_CENTER
        )


        # calculate the detections position inside the target RoI
        transformed_points = view_transformer.transform_points(points=points).astype(int)


        # store detections position and historical positions and class IDs
        current_positions = {}
        current_classes = {} # Dictionary to store class IDs for each tracked object
        for i, (tracker_id, point) in enumerate(zip(detections.tracker_id, transformed_points)):
            coordinates[tracker_id].append(point[1]) # Store y-coordinate for speed calculation (in transformed space)
            historical_positions[tracker_id].append(point) # Store transformed point for distance and direction calculation
            current_positions[tracker_id] = point
            current_classes[tracker_id] = detections.class_id[i] # Store the class ID for the current frame


            # Calculate and store current speed for evasive action inference
            if len(coordinates[tracker_id]) > 1:
                coordinate_start = coordinates[tracker_id][-2]
                coordinate_end = coordinates[tracker_id][-1]
                distance_y = abs(coordinate_start - coordinate_end)
                time = 1 / video_info.fps # Time between last two frames
                current_speed = (distance_y / time) * 3.6 # Speed in km/h
                historical_speeds[tracker_id].append(current_speed)
            else:
                historical_speeds[tracker_id].append(0) # Append 0 for the first frame


        # Calculate distances and relative speeds between pairs of objects
        conflict_data_frame = [] # List to store conflict data for the current frame
        object_ids = list(current_positions.keys())
        for i in range(len(object_ids)):
            for j in range(i + 1, len(object_ids)):
                id1 = object_ids[i]
                id2 = object_ids[j]


                if len(historical_positions[id1]) > 1 and len(historical_positions[id2]) > 1:
                    # Calculate distance
                    distance = np.linalg.norm(current_positions[id1] - current_positions[id2])


                    # Calculate individual velocities from historical positions
                    time_interval = 1 / video_info.fps
                    # Ensure we have enough history for velocity calculation
                    if len(historical_positions[id1]) > 1:
                         velocity1 = (historical_positions[id1][-1] - historical_positions[id1][-2]) / time_interval
                    else:
                         velocity1 = np.array([0, 0]) # Default to zero velocity if not enough history


                    if len(historical_positions[id2]) > 1:
                        velocity2 = (historical_positions[id2][-1] - historical_positions[id2][-2]) / time_interval
                    else:
                        velocity2 = np.array([0, 0]) # Default to zero velocity if not enough history


                    # Calculate relative velocity and speed
                    relative_velocity = velocity1 - velocity2
                    relative_speed = np.linalg.norm(relative_velocity)


                    # Conflict detection
                    if distance < 5 and relative_speed > 0: # Distance threshold 5m and relative speed > 0
                        # Calculate TTC
                        # Project the relative velocity onto the vector connecting the two objects
                        # This gives the closing speed
                        vector_between = current_positions[id2] - current_positions[id1]
                        closing_speed = -np.dot(relative_velocity, vector_between) / distance if distance > 0 else 0 # Avoid division by zero


                        if closing_speed > 0: # Only calculate TTC if objects are closing in
                            ttc = distance / closing_speed
                        else:
                            ttc = float('inf') # No risk of collision if not closing in


                        # Determine severity
                        if ttc < 1.5:
                            severity = 'severe'
                        elif 1.5 <= ttc < 3:
                            severity = 'non-severe'
                        else:
                            severity = 'no conflict' # Should not be triggered with the initial conflict condition, but good for completeness


                        # Determine conflict type (simplified for rear-end and crossing)
                        conflict_type = 'unknown'
                        # A simplified approach: if vehicles are moving in roughly the same direction, it's rear-end. Otherwise, crossing.
                        # This is a basic approximation and might need refinement based on specific road layout and traffic patterns.
                        dot_product = np.dot(velocity1, velocity2)
                        if dot_product > 0.5 * np.linalg.norm(velocity1) * np.linalg.norm(velocity2) and np.linalg.norm(velocity1) > movement_threshold and np.linalg.norm(velocity2) > movement_threshold: # Check if velocities are roughly in the same direction and vehicles are moving
                            conflict_type = 'rear-end'
                        elif np.linalg.norm(velocity1) > movement_threshold and np.linalg.norm(velocity2) > movement_threshold: # Check if both vehicles are moving
                             conflict_type = 'crossing'
                        else:
                            conflict_type = 'unknown'


                        # Determine direction of movement for each object
                        direction1 = determine_direction(historical_positions[id1])
                        direction2 = determine_direction(historical_positions[id2])


                        # Infer evasive action for each object
                        evasive_action1 = infer_evasive_action(historical_speeds[id1])
                        evasive_action2 = infer_evasive_action(historical_speeds[id2])


                        # Get class names for the involved objects
                        class_name1 = model.names[current_classes[id1]] if id1 in current_classes else 'unknown'
                        class_name2 = model.names[current_classes[id2]] if id2 in current_classes else 'unknown'


                        # Store conflict data for the current frame
                        if severity != 'no conflict':
                            # Calculate individual speeds for the conflict data
                            speed1 = np.linalg.norm(velocity1) * 3.6 # Convert to km/h
                            speed2 = np.linalg.norm(velocity2) * 3.6 # Convert to km/h


                            conflict_data_frame.append({
                                'frame-id': frame_idx,
                                'timestamp': timestamp_formatted,  # Use the formatted timestamp here
                                'type of conflict': conflict_type,
                                'user-id A': id1,
                                'user-id B': id2,
                                'class A': class_name1, # Add class name for object A
                                'class B': class_name2, # Add class name for object B
                                'speed A (km/h)': speed1,
                                'speed B (km/h)': speed2,
                                'TTC': ttc,
                                'severity': severity,
                                'distance AB': distance,
                                'relative speed (m/s)': relative_speed, # Keep relative speed in m/s for consistency with velocity calculations
                                'direction A': direction1, # Add direction for object A
                                'direction B': direction2, # Add direction for object B
                                'evasive action A': evasive_action1, # Add evasive action for object A
                                'evasive action B': evasive_action2 # Add evasive action for object B
                            })


        # Create a DataFrame for the current frame's conflict data
        conflict_df_frame = pd.DataFrame(conflict_data_frame)


        # Export conflict data for the current frame to Excel (append mode)
        # If file exists, append without header, otherwise create with header
        mode = 'a' if os.path.exists(excel_file_path) else 'w'
        header = not os.path.exists(excel_file_path) # Write header only if file does not exist


        with pd.ExcelWriter(excel_file_path, mode=mode, engine='openpyxl', if_sheet_exists='overlay' if mode == 'a' else None) as writer:
            conflict_df_frame.to_excel(writer, index=False, header=header)


        # format labels
        labels = []


        for tracker_id in detections.tracker_id:
            if len(coordinates[tracker_id]) < video_info.fps / 2:
                labels.append(f"#{tracker_id}")
            else:
                # calculate speed
                coordinate_start = coordinates[tracker_id][-1]
                coordinate_end = coordinates[tracker_id][0]
                # Calculate speed based on distance covered over a longer period for more stability
                distance_y = abs(coordinate_start - coordinate_end)
                time = len(coordinates[tracker_id]) / video_info.fps # Time duration over which distance is measured
                speed = distance_y / time * 3.6 # Speed in km/h


                # Get class name for the label
                class_name = model.names[current_classes[tracker_id]] if tracker_id in current_classes else 'unknown'


                labels.append(f"#{tracker_id} {class_name} {int(speed)} km/h")


        # annotate frame
        annotated_frame = frame.copy()
        annotated_frame = trace_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = bounding_box_annotator.annotate(
            scene=annotated_frame, detections=detections
        )
        annotated_frame = label_annotator.annotate(
            scene=annotated_frame, detections=detections, labels=labels
        )


        # add frame to target video
        sink.write_frame(annotated_frame)


# After the loop, move the generated excel file to the specified drive location
import shutil
import os


# Ensure the target directory exists in Google Drive
drive_output_dir = "/content/drive/MyDrive/TrafficConflictAnalysis"
os.makedirs(drive_output_dir, exist_ok=True)


# Copy the result video to Google Drive
# shutil.copy(TARGET_VIDEO_PATH, drive_output_dir) # Uncomment if you want to copy the video
# print(f"Result video saved to: {drive_output_dir}/{os.path.basename(TARGET_VIDEO_PATH)}")


# Assuming the excel file will be named 'traffic_conflicts.xlsx' and saved in the current directory
excel_file_path = 'traffic_conflicts.xlsx'
drive_excel_path = os.path.join(drive_output_dir, 'succesa.xlsx') # Changed filename to succesa.xlsx


if os.path.exists(excel_file_path):
    shutil.copy(excel_file_path, drive_excel_path) # Copy with the new filename
    print(f"Traffic conflict data saved to: {drive_excel_path}")
else:
    print(f"Excel file not found at {excel_file_path}. Please ensure the file is generated before attempting to save.")


