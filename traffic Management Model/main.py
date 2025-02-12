import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*

model = YOLO('yolov8s.pt')

# Vehicle-related classes (as per YOLO)
vehicle_classes = ['car', 'motorcycle', 'bus', 'truck']

# Emergency vehicles (Checking detected names)
emergency_classes = ['ambulance', 'fire truck', 'police', 'truck', 'bus']  # YOLO may detect as truck/bus

tracker = Tracker()
count = 0

cap = cv2.VideoCapture('traffic_with_emergency.mp4')

down = {}
up = {}
counter_down = []
counter_up = []
emergency_crossed_red = False  # Track if an emergency vehicle has crossed the red line
emergency_crossed_blue = False  # Track if an emergency vehicle has crossed the blue line

while True:    
    ret, frame = cap.read()
    if not ret:
        break
    
    count += 1
    frame = cv2.resize(frame, (1020, 500))

    results = model.predict(frame)
    detections = results[0].boxes.data.detach().cpu().numpy()
    px = pd.DataFrame(detections).astype("float")

    detected_vehicles = []
    for _, row in px.iterrows():
        x1, y1, x2, y2, _, class_id = map(int, row[:6])
        vehicle_type = model.names[class_id]  # Get detected class name
        
        if vehicle_type in vehicle_classes or vehicle_type in emergency_classes:
            detected_vehicles.append([x1, y1, x2, y2, vehicle_type])  # Store vehicle type

    bbox_id = tracker.update([box[:4] for box in detected_vehicles])  # Track using only bbox coords

    for i, bbox in enumerate(bbox_id):
        x3, y3, x4, y4, obj_id = bbox
        cx = int((x3 + x4) / 2)
        cy = int((y3 + y4) / 2)

        red_line_y = 198
        blue_line_y = 400
        offset = 7

        # Red line logic
        if red_line_y - offset < cy < red_line_y + offset:
            down[obj_id] = cy   

        if obj_id in down:
            cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(frame, str(obj_id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
            counter_down.append(obj_id)

            # Check if the vehicle that crossed is an emergency vehicle
            if detected_vehicles[i][4] in emergency_classes:
                emergency_crossed_red = True

        # Blue line logic
        if blue_line_y - offset < cy < blue_line_y + offset:
            up[obj_id] = cy   

        if obj_id in up:
            cv2.circle(frame, (cx, cy), 4, (255, 0, 0), -1)
            cv2.putText(frame, str(obj_id), (cx, cy), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
            counter_up.append(obj_id)

            # Check if the vehicle that crossed is an emergency vehicle
            if detected_vehicles[i][4] in emergency_classes:
                emergency_crossed_blue = True

        # Display vehicle name on top
        vehicle_name = detected_vehicles[i][4]  # Get vehicle type
        cv2.putText(frame, vehicle_name, (x3, y3 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    total_vehicles = len(set(counter_down))
    
    # Signal Logic: Green if emergency vehicle crossed red line, Red if it crossed blue line
    if emergency_crossed_red and not emergency_crossed_blue:
        signal_color = (0, 255, 0)  # Green
    elif total_vehicles > 50:  # Additional condition: Green if vehicle count > 30
        signal_color = (0, 255, 0)  # Green
    else:
        signal_color = (0, 0, 255)  # Red

    # Draw signal at the center (No Text)
    cv2.rectangle(frame, (470, 20), (520, 70), signal_color, -1)

    # Draw red line
    cv2.line(frame, (222, red_line_y), (824, red_line_y), (0, 0, 255), 3)
    cv2.putText(frame, 'Red Line', (222, red_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Draw blue line
    cv2.line(frame, (172, blue_line_y), (774, blue_line_y), (255, 0, 0), 3)
    cv2.putText(frame, 'Blue Line', (172, blue_line_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)

    # Vehicle count display
    cv2.putText(frame, f'Vehicles: {total_vehicles}', (60, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("Traffic Monitoring", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()