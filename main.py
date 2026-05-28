from keras_facenet import FaceNet
import cv2  
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import os
import torch
import threading
import time
from ultralytics import YOLO
from datetime import datetime
import csv


embedder = FaceNet()
model = YOLO("yolov8n-face.pt", task="detect")
device = 0 if torch.cuda.is_available() else "cpu"
similarity_threshold = 0.60

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
photos_path = os.path.join(BASE_DIR, "data")


def prepare_registered_face(image, image_name):
    results = model(image, device=device, verbose=False)
    best_box = None
    best_area = 0

    for result in results:
        for box in result.boxes:
            confidence = float(box.conf[0])
            if confidence < 0.4:
                continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_box = (x1, y1, x2, y2)

    if best_box is None:
        print(f"Warning: face not detected in {image_name}, using full image.")
        face = image
    else:
        x1, y1, x2, y2 = best_box
        face = image[y1:y2, x1:x2]

    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = cv2.resize(face, (160, 160)).astype(np.float32)
    return face


users_face = sorted(os.listdir(photos_path))
all_users_face = []
all_users_face_name = []
for user in users_face:
    person_image = cv2.imread(os.path.join(photos_path, user))
    if person_image is None:
        print(f"Warning: cannot read {user}, skipped.")
        continue

    person_face = prepare_registered_face(person_image, user)
    all_users_face.append(person_face)
    all_users_face_name.append(os.path.splitext(user)[0])

all_users_face = np.array(all_users_face)
if len(all_users_face) == 0:
    raise RuntimeError("No registered face images found in data folder.")

face_embeddings = embedder.embeddings(all_users_face)
print(f"Loaded registered faces: {', '.join(all_users_face_name)}")


class LatestFrameCapture:
    def __init__(self, source):
        self.cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.lock = threading.Lock()
        self.ret = False
        self.frame = None
        self.running = self.cap.isOpened()
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.05)
                continue
            with self.lock:
                self.ret = ret
                self.frame = frame

    def is_opened(self):
        return self.running and self.cap.isOpened()

    def read(self):
        with self.lock:
            if self.frame is None:
                return False, None
            return self.ret, self.frame.copy()

    def release(self):
        self.running = False
        self.thread.join(timeout=1)
        self.cap.release()


def detect_faces(frame, detect_width):
    frame_height, frame_width = frame.shape[:2]
    scale = min(1.0, detect_width / frame_width)
    detect_frame = cv2.resize(frame, (int(frame_width * scale), int(frame_height * scale)))

    results = model(detect_frame, device=device, verbose=False)
    all_faces = []
    face_cordinates = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = box.conf[0]
            if confidence > 0.5:
                if scale != 1.0:
                    x1 = int(x1 / scale)
                    y1 = int(y1 / scale)
                    x2 = int(x2 / scale)
                    y2 = int(y2 / scale)
                croped_frame = frame[y1:y2, x1:x2]
                if croped_frame.size == 0:
                    continue
                croped_frame = cv2.cvtColor(croped_frame, cv2.COLOR_BGR2RGB)
                croped_frame = cv2.resize(croped_frame, (160, 160)).astype(np.float32)
                all_faces.append(croped_frame)
                face_cordinates.append((x1, y1, x2, y2))

    detections = []
    if len(all_faces) > 0:
        cctv_face_embeddings = embedder.embeddings(np.array(all_faces))
        similarities = cosine_similarity(face_embeddings, cctv_face_embeddings)
        best_user_indexes = similarities.argmax(axis=0)

        for face_index, user_index in enumerate(best_user_indexes):
            x1, y1, x2, y2 = face_cordinates[face_index]
            similarity = similarities[user_index, face_index]

            if similarity > similarity_threshold:
                name = all_users_face_name[user_index]
                current_time = datetime.now().strftime("%H:%M:%S")

                if name not in session_attendance:
                    session_attendance[name] = {"in": current_time}

                display_text = name
                box_color = (0, 255, 0)
            else:
                display_text = ""
                box_color = (0, 255, 255)

            detections.append((x1, y1, x2, y2, display_text, box_color))

    return detections


class DetectionWorker:
    def __init__(self, detect_width):
        self.detect_width = detect_width
        self.frame_lock = threading.Lock()
        self.result_lock = threading.Lock()
        self.frame = None
        self.detections = []
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def update_frame(self, frame):
        with self.frame_lock:
            self.frame = frame.copy()

    def get_detections(self):
        with self.result_lock:
            return list(self.detections)

    def _run(self):
        while self.running:
            with self.frame_lock:
                frame = None if self.frame is None else self.frame.copy()

            if frame is None:
                time.sleep(0.01)
                continue

            detections = detect_faces(frame, self.detect_width)
            with self.result_lock:
                self.detections = detections

    def stop(self):
        self.running = False
        self.thread.join(timeout=1)


session_attendance = {}
today = datetime.now().strftime("%Y-%m-%d")
csv_filename = f"attendance_{today}.csv"

# Connect to RTSP stream.
# Channel 102 is usually the lower-resolution sub-stream. Use 101 if you need the main stream.
rtsp_url = "rtsp://admin:@dmin456@192.168.1.69:554/Streaming/Channels/102"
cap = LatestFrameCapture(rtsp_url)
detect_width = 480
detector = DetectionWorker(detect_width)
window_name = "YOLOv8 Face Detection"
fullscreen = True

cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
if fullscreen:
    cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

while cap.is_opened():
    ret, frame = cap.read()
    if not ret:
        print("Waiting for CCTV frame ...")
        time.sleep(0.05)
        continue

    detector.update_frame(frame)

    for x1, y1, x2, y2, display_text, box_color in detector.get_detections():
        cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
        if display_text:
            cv2.putText(
                frame,
                display_text,
                (x1, y1),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                box_color,
                2
            )

    cv2.imshow(window_name, frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

detector.stop()
cap.release()
cv2.destroyAllWindows()

with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["Name", "Time"])
    for name, times in session_attendance.items():
        writer.writerow([name, times["in"] ])
