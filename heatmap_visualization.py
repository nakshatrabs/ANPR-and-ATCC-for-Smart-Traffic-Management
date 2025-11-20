import cv2
import numpy as np
from ultralytics import YOLO
import os

def generate_heatmap(frame, vehicle_positions, intensity=15):
    """
    Generate a heatmap of vehicle positions and overlay it on the frame.
    """
    heatmap = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.float32)
    
    for x, y in vehicle_positions:
        if 0 <= int(y) < frame.shape[0] and 0 <= int(x) < frame.shape[1]:
            heatmap[int(y), int(x)] += 1
    
    if heatmap.max() > 0:
        heatmap = cv2.GaussianBlur(heatmap, (0, 0), intensity)
        heatmap = np.clip(heatmap / heatmap.max(), 0, 1) * 255
        heatmap_colored = cv2.applyColorMap(heatmap.astype(np.uint8), cv2.COLORMAP_JET)
        overlayed_frame = cv2.addWeighted(frame, 0.6, heatmap_colored, 0.4, 0)
        return overlayed_frame
    else:
        return frame

def track_vehicle_movement(vehicle_centers, prev_centers):
    """
    Draw arrows to show vehicle movement direction.
    """
    movement_vectors = []
    for i, (cx, cy) in enumerate(vehicle_centers):
        if i < len(prev_centers):
            px, py = prev_centers[i]
            movement_vectors.append(((int(px), int(py)), (int(cx), int(cy))))
    return movement_vectors

def determine_signal(total_vehicles):
    """Determine the signal color based on vehicle count."""
    if total_vehicles < 10:
        return "Green", (0, 255, 0)
    elif total_vehicles < 20:
        return "Yellow", (0, 255, 255)
    else:
        return "Red", (0, 0, 255)

def process_frame(frame, model, road_name, directions, prev_positions):
    """
    Process a video frame: Detect objects, classify directions, and overlay information.
    """
    results = model(frame)
    detections = results[0].boxes.data.cpu().numpy()
    vehicle_count = {"car": 0, "truck": 0, "motorcycle": 0, "bus": 0}
    vehicle_positions = []

    for det in detections:
        x1, y1, x2, y2, conf, class_id = det
        label = model.names[int(class_id)]
        
        if label in vehicle_count:
            vehicle_count[label] += 1
            center_x = (x1 + x2) / 2
            center_y = (y1 + y2) / 2
            vehicle_positions.append((center_x, center_y))

            if center_x < frame.shape[1] / 2:
                directions['left'] += 1
            else:
                directions['right'] += 1

            color = (0, 255, 0) if conf > 0.5 else (0, 0, 255)
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            cv2.putText(frame, f"{label} {conf:.2f}", (int(x1), int(y1) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Add heatmap
    frame = generate_heatmap(frame, vehicle_positions)

    # Add movement arrows
    movement_vectors = track_vehicle_movement(vehicle_positions, prev_positions)
    for start, end in movement_vectors:
        cv2.arrowedLine(frame, start, end, (0, 255, 0), 2)

    # Display info
    cv2.putText(frame, f"Road: {road_name}", (10, 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    y_offset = 50
    for direction, count in directions.items():
        cv2.putText(frame, f"{direction.capitalize()}: {count}", (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y_offset += 25

    total_vehicles = sum(vehicle_count.values())
    cv2.putText(frame, f"Total Vehicles: {total_vehicles}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    y_offset += 25

    signal_color, signal_rgb = determine_signal(total_vehicles)
    cv2.putText(frame, f"Signal: {signal_color}", (10, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, signal_rgb, 2)

    return frame, vehicle_count, vehicle_positions

def process_videos(video_files, model):
    """
    Process a list of video file paths and display results.
    """
    if isinstance(video_files, str):
        # If a single string is passed, wrap it into a list
        video_files = [video_files]

    caps, road_names, prev_positions_list = [], [], []

    for video_path in video_files:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video file: {video_path}")
            continue
        caps.append(cap)
        road_names.append(os.path.basename(video_path))
        prev_positions_list.append([])

    target_width, target_height = 640, 480

    while True:
        processed_frames = []
        for i, cap in enumerate(caps):
            ret, frame = cap.read()
            if not ret:
                print(f"End of video stream for {road_names[i]}.")
                caps[i].release()
                continue

            directions = {"left": 0, "right": 0}
            frame_resized = cv2.resize(frame, (target_width, target_height))

            processed_frame, vehicle_count, vehicle_positions = process_frame(
                frame_resized, model, road_names[i], directions, prev_positions_list[i]
            )

            prev_positions_list[i] = vehicle_positions
            processed_frames.append(processed_frame)

        if len(processed_frames) == 0:
            break

        # Combine frames horizontally (if multiple)
        if len(processed_frames) > 1:
            grid_frame = np.hstack(processed_frames)
        else:
            grid_frame = processed_frames[0]

        cv2.imshow("Traffic Heatmap Visualization", grid_frame)

        if cv2.waitKey(20) & 0xFF == ord('q'):
            break

    for cap in caps:
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = r"C:\Users\hp\OneDrive\Desktop\Projects\Infosys SpringBoard Internship\ANPR-and-ATCC-for-Smart-Traffic-Management-main\sample detection videos\triple riding.mp4"
                  
    model = YOLO("yolov8n.pt")
    process_videos(video_path, model)