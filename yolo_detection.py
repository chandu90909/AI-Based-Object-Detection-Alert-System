import cv2
from ultralytics import YOLO

# Load YOLO model
model = YOLO("yolov8n.pt")

# 📱 Use mobile camera (DroidCam)
cap = cv2.VideoCapture(1)   # change to 2 if not working

while True:
    success, img = cap.read()
    if not success:
        break

    # Run YOLO detection
    results = model(img)

    person_count = 0
    phone_detected = False

    # Process detections
    for r in results:
        for box in r.boxes:
            cls = int(box.cls[0])
            label = model.names[cls]

            x1, y1, x2, y2 = map(int, box.xyxy[0])

            # 🎯 Track ONLY person (draw box manually)
            if label == "person":
                person_count += 1
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 3)
                cv2.putText(img, "PERSON", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 🚨 Detect phone
            if label == "cell phone":
                phone_detected = True
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                cv2.putText(img, "PHONE", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # 📊 Show counts
    cv2.putText(img, f'Persons: {person_count}', (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    # 🚨 Alert message
    if phone_detected:
        cv2.putText(img, "PHONE DETECTED!", (50, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 4)

    cv2.imshow("YOLO Mobile Camera Detection", img)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()