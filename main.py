import cv2
import pandas as pd
import numpy as np
from ultralytics import YOLO
import time

model = YOLO('yolov8s.pt')

def RGB(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE:
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('RGB')
cv2.setMouseCallback('RGB', RGB)

cap = cv2.VideoCapture('parking1.mp4')

my_file = open("coco.txt", "r")
data = my_file.read()
class_list = data.split("\n")

# Define parking areas
areas = [
    [(52, 364), (30, 417), (73, 412), (88, 369)],
    [(105, 353), (86, 428), (137, 427), (146, 358)],
    [(159, 354), (150, 427), (204, 425), (203, 353)],
    [(217, 352), (219, 422), (273, 418), (261, 347)],
    [(274, 345), (286, 417), (338, 415), (321, 345)],
    [(336, 343), (357, 410), (409, 408), (382, 340)],
    [(396, 338), (426, 404), (479, 399), (439, 334)],
    [(458, 333), (494, 397), (543, 390), (495, 330)],
    [(511, 327), (557, 388), (603, 383), (549, 324)],
    [(564, 323), (615, 381), (654, 372), (596, 315)],
    [(616, 316), (666, 369), (703, 363), (642, 312)],
    [(674, 311), (730, 360), (764, 355), (707, 308)]
]

# Initialize variables
is_playing = False  # To toggle play/pause
paused = False  # To keep track of pause state

print("Press SPACE to play/pause the video, or ESC to exit.")

# Main loop to handle video play/pause with space key
while True:
    key = cv2.waitKey(1) & 0xFF

    if key == 32:  # Space key
        is_playing = not is_playing  # Toggle play/pause
        paused = not paused  # Toggle the paused state

    elif key == 27:  # Escape key to exit
        cap.release()
        cv2.destroyAllWindows()
        break

    if is_playing:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, (1020, 500))

        results = model.predict(frame)
        a = results[0].boxes.data
        px = pd.DataFrame(a).astype("float")

        slot_lists = [[] for _ in range(12)]

        for index, row in px.iterrows():
            x1 = int(row[0])
            y1 = int(row[1])
            x2 = int(row[2])
            y2 = int(row[3])
            d = int(row[5])
            c = class_list[d]

            if 'car' in c:
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                for i, area in enumerate(areas):
                    result = cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False)
                    if result >= 0:
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                        slot_lists[i].append(c)
                        cv2.putText(frame, str(c), (x1, y1), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 255, 255), 1)

        # Determine free and busy slots
        free_slots = []
        busy_slots = []

        for i, slot_list in enumerate(slot_lists):
            if len(slot_list) > 0:
                busy_slots.append(i + 1)  # Add slot number to busy list
            else:
                free_slots.append(i + 1)  # Add slot number to free list

        # Draw parking slot areas with numbers
        for i, area in enumerate(areas):
            color = (0, 0, 255) if i + 1 in busy_slots else (0, 255, 0)
            cv2.polylines(frame, [np.array(area, np.int32)], True, color, 2)
            cv2.putText(frame, str(i + 1), (area[0][0], area[0][1] + 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, color, 1)

        # Display free and busy slots
        free_text = "Free Slots: " + ", ".join(map(str, free_slots)) if free_slots else "Free Slots: None"
        busy_text = "Busy Slots: " + ", ".join(map(str, busy_slots)) if busy_slots else "Busy Slots: None"

        cv2.putText(frame, free_text, (23, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
        cv2.putText(frame, busy_text, (23, 60), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 0, 255), 2)

        # Show frame
        cv2.imshow("RGB", frame)

    # If paused, wait for spacebar press to resume
    if paused:
        time.sleep(0.1)  # Keep waiting for key press while paused

cap.release()
cv2.destroyAllWindows()
