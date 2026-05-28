from keras_facenet import FaceNet
import cv2  
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import torch
from ultralytics import YOLO
from datetime import datetime
import csv


embedder = FaceNet()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
photos_path = os.path.join(BASE_DIR, "data")
users_face = os.listdir(photos_path)
all_users_face = []
all_users_face_name = []
for user in users_face:
    person_image = cv2.imread(os.path.join(photos_path, user))
    person_rgb_image = cv2.cvtColor(person_image, cv2.COLOR_BGR2RGB)
    person_resize_image = cv2.resize(person_rgb_image, (160, 160)).astype(np.float32)
    all_users_face.append(person_resize_image)
    all_users_face_name.append(user.split('.')[0])
all_users_face = np.array(all_users_face)


face_embeddings = embedder.embeddings(all_users_face)


model = YOLO("yolov8n-face.pt", task="detect")


session_attendance = {}
today = datetime.now().strftime("%Y-%m-%d")
csv_filename = f"attendance_{today}.csv"

# Connect to RTSP stream 
rtsp_url = "rtsp://admin:@dmin456@192.168.18.3:554/Streaming/Channels/101"
cap = cv2.VideoCapture(rtsp_url)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break  

    results = model(frame, device='cpu') 
    all_faces = []
    face_cordinates = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            if confidence > 0.5:
                croped_frame = frame[y1:y2, x1:x2]
                croped_frame = cv2.cvtColor(croped_frame, cv2.COLOR_BGR2RGB)
                croped_frame = cv2.resize(croped_frame, (160, 160)).astype(np.float32)
                all_faces.append(croped_frame)
                face_cordinates.append((x1, y1, x2, y2))

    if len(all_faces) > 0:
        cctv_face_embeddings = embedder.embeddings(np.array(all_faces))
        similarities = cosine_similarity(face_embeddings, cctv_face_embeddings)
        best_user_indexes = similarities.argmax(axis=0)

        for face_index, user_index in enumerate(best_user_indexes):
            x1, y1, x2, y2 = face_cordinates[face_index]
            similarity = similarities[user_index, face_index]

            if similarity > 0.65:
                name = all_users_face_name[user_index]
                current_time = datetime.now().strftime("%H:%M:%S")

                
                if name not in session_attendance:
                    session_attendance[name] = {"in": current_time, }
                

                accuracy = similarity * 100

                display_text = f"{name} ({accuracy:.2f}%)"
                box_color = (0, 255, 0)
            else:
                display_text = "Data not found"
                box_color = (0, 255, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            cv2.putText(
                frame,
                display_text,
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                box_color,
                2
            )

    cv2.imshow('YOLOv8 Face Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Time"])
    for name, times in session_attendance.items():
        writer.writerow([name, times["in"] ])
