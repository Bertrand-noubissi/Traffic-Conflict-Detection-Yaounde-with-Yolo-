!nvidia-smi


import os
HOME = os.getcwd()
print(HOME)





!pip install ultralytics==8.3.19


# prevent ultralytics from tracking your activity
!yolo settings sync=False


from IPython import display
display.clear_output()


import ultralytics
ultralytics.checks()



!pip install supervision[assets]==0.24.0


from IPython import display
display.clear_output()


import supervision as sv
print("supervision.__version__:", sv.__version__)



SOURCE_VIDEO_PATH ='/content/drive/MyDrive/Acasia.mp4'


from ultralytics import YOLO


model = YOLO("yolov8x.pt")


# dict maping class_id to class_name
CLASS_NAMES_DICT = model.model.names


# the class names we have chosen
SELECTED_CLASS_NAMES = ['car', 'motorcycle', 'bus', 'truck', 'person']


# class ids matching the class names we have chosen
SELECTED_CLASS_IDS = [
    {value: key for key, value in CLASS_NAMES_DICT.items()}[class_name]
    for class_name
    in SELECTED_CLASS_NAMES
]





import supervision as sv
import numpy as np




# create frame generator
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)
# create instance of BoxAnnotator and LabelAnnotator
box_annotator = sv.BoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5, text_color=sv.Color.BLACK)
# acquire first video frame
iterator = iter(generator)
frame = next(iterator)
# model prediction on single frame and conversion to supervision Detections
results = model(frame, verbose=False)[0]


# convert to Detections
detections = sv.Detections.from_ultralytics(results)
# only consider class id from selected_classes define above
detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]


# format custom labels
labels = [
    f"{CLASS_NAMES_DICT[class_id]} {confidence:0.2f}"
    for confidence, class_id in zip(detections.confidence, detections.class_id)
]


# annotate and display frame
annotated_frame = frame.copy()
annotated_frame = box_annotator.annotate(
    scene=annotated_frame, detections=detections)
annotated_frame = label_annotator.annotate(
    scene=annotated_frame, detections=detections, labels=labels)


%matplotlib inline
sv.plot_image(annotated_frame, (16, 16))



# settings
LINE_START = sv.Point(50, 410)
LINE_END = sv.Point(400, 410)


LINE_START_2 = sv.Point(55, 410)
LINE_END_2 = sv.Point(350, 710)


LINE_START_3 = sv.Point(350, 700)
LINE_END_3 = sv.Point(700, 560)


LINE_START_4 = sv.Point(400, 400)
LINE_END_4 = sv.Point(700, 550)


TARGET_VIDEO_PATH = f"{HOME}/result.mp4"



sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)




# create BYTETracker instance
byte_tracker = sv.ByteTrack(
    track_activation_threshold=0.25,
    lost_track_buffer=30,
    minimum_matching_threshold=0.8,
    frame_rate=30,
    minimum_consecutive_frames=3)


byte_tracker.reset()


# create VideoInfo instance
video_info = sv.VideoInfo.from_video_path(SOURCE_VIDEO_PATH)


# create frame generator
generator = sv.get_video_frames_generator(SOURCE_VIDEO_PATH)


# create LineZone instance, it is previously called LineCounter class
line_zone = sv.LineZone(start=LINE_START, end=LINE_END)
line_zone_2 = sv.LineZone(start=LINE_START_2, end=LINE_END_2)
line_zone_3 = sv.LineZone(start=LINE_START_3, end=LINE_END_3)
line_zone_4 = sv.LineZone(start=LINE_START_4, end=LINE_END_4)


# Initialize dictionaries to store counts per category for each line zone
line_zone_category_counts_in = {i: {class_name: 0 for class_name in SELECTED_CLASS_NAMES} for i in range(4)}
line_zone_category_counts_out = {i: {class_name: 0 for class_name in SELECTED_CLASS_NAMES} for i in range(4)}


# create instance of BoxAnnotator, LabelAnnotator, and TraceAnnotator
box_annotator = sv.BoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_thickness=2, text_scale=0.5, text_color=sv.Color.BLACK)
trace_annotator = sv.TraceAnnotator(thickness=4, trace_length=50)


# create LineZoneAnnotator instance, it is previously called LineCounterAnnotator class
line_zone_annotator = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=1)
line_zone_annotator_2 = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=1)
line_zone_annotator_3 = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=1)
line_zone_annotator_4 = sv.LineZoneAnnotator(thickness=4, text_thickness=4, text_scale=1)


# define call back function to be used in video processing
def callback(frame: np.ndarray, index: int) -> tuple[np.ndarray, dict]:
    global line_zone_category_counts_in, line_zone_category_counts_out
    # model prediction on single frame and conversion to supervision Detections
    results = model(frame, verbose=False)[0]
    detections = sv.Detections.from_ultralytics(results)
    # only consider class id from selected_classes define above
    detections = detections[np.isin(detections.class_id, SELECTED_CLASS_IDS)]
    # tracking detections
    detections = byte_tracker.update_with_detections(detections)


    # Build labels for annotation - only for detections with a tracker_id
    labels = []
    # Filter detections to only include those with a valid tracker_id for annotation and triggering
    tracked_detections = detections[detections.tracker_id is not None]


    if tracked_detections.tracker_id is not None and len(tracked_detections.tracker_id) > 0: # Check if tracked_detections is not empty after filtering
        labels = [
            f"#{tracker_id} {model.model.names[class_id]} {confidence:0.2f}"
            for confidence, class_id, tracker_id
            in zip(tracked_detections.confidence, tracked_detections.class_id, tracked_detections.tracker_id)
        ]


    annotated_frame = frame.copy()
    # Annotate using only tracked detections
    annotated_frame = trace_annotator.annotate(
        scene=annotated_frame, detections=tracked_detections)
    annotated_frame = box_annotator.annotate(
        scene=annotated_frame, detections=tracked_detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=tracked_detections, labels=labels)


    # update line counters and category counts
    line_zones = [line_zone, line_zone_2, line_zone_3, line_zone_4]
    for i, line_zone_instance in enumerate(line_zones):
        # Get the detections that triggered the line zone in this frame using tracked_detections
        triggers = line_zone_instance.trigger(tracked_detections)


        # Update category counts based on triggers
        if triggers is not None and len(triggers) > 0:
            # Ensure triggers is a list of tuples with at least two elements
            valid_triggers = [t for t in triggers if isinstance(t, (list, tuple)) and len(t) > 1]
            for trigger_detection in valid_triggers:
                class_id = trigger_detection[1]
                class_name = CLASS_NAMES_DICT[class_id]
                # Determine direction (this requires more complex logic based on previous position/movement)
                # For simplicity here, we'll increment both in and out for any trigger,
                # you'd need to implement direction logic for accurate counts.
                line_zone_category_counts_in[i][class_name] += 1
                line_zone_category_counts_out[i][class_name] += 1


    # Store counts for each line and category
    counts_data = {}
    for i, line_zone_instance in enumerate(line_zones):
        line_number = i + 1
        counts_data[f'line_{line_number}_in_total'] = line_zone_instance.in_count
        counts_data[f'line_{line_number}_out_total'] = line_zone_instance.out_count


        # Add category counts
        for class_name in SELECTED_CLASS_NAMES:
             counts_data[f'line_{line_number}_in_{class_name}'] = line_zone_category_counts_in[i][class_name]
             counts_data[f'line_{line_number}_out_{class_name}'] = line_zone_category_counts_out[i][class_name]


    # return frame with box and line annotated result
    annotated_frame = line_zone_annotator.annotate(annotated_frame, line_counter=line_zone)
    annotated_frame = line_zone_annotator_2.annotate(annotated_frame, line_counter=line_zone_2)
    annotated_frame = line_zone_annotator_3.annotate(annotated_frame, line_counter=line_zone_3)
    annotated_frame = line_zone_annotator_4.annotate(annotated_frame, line_counter=line_zone_4)


    return annotated_frame, counts_data


# process the whole video
# Modified to handle the tuple return from the callback function
def process_video_with_counts(source_path: str, target_path: str, callback: callable) -> list:
    all_counts_data = []
    with sv.VideoSink(target_path=target_path, video_info=sv.VideoInfo.from_video_path(source_path)) as sink:
        for index, frame in enumerate(sv.get_video_frames_generator(source_path)):
            annotated_frame, counts_data = callback(frame, index)
            sink.write_frame(frame=annotated_frame)
            all_counts_data.append(counts_data)
    return all_counts_data


all_frame_counts = process_video_with_counts(
    source_path = SOURCE_VIDEO_PATH,
    target_path = TARGET_VIDEO_PATH,
    callback=callback
)






import pandas as pd
from IPython.display import display


counts_df = pd.DataFrame(all_frame_counts)
display(counts_df.head())





drive_excel_output_path = '/content/drive/MyDrive/line_counts.xlsx'
counts_df.to_excel(drive_excel_output_path, index=False)


print(f"Counting results exported to Google Drive at {drive_excel_output_path}")

