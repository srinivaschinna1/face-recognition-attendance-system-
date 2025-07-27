import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

# Load face detector and embedding model
face_detector = cv2.dnn.readNetFromCaffe(
    "models/deploy.prototxt",
    "models/res10_300x300_ssd_iter_140000.caffemodel"
)
face_embedder = cv2.dnn.readNetFromTorch("models/openface.nn4.small2.v1.t7")

# Attendance Logger
def mark_attendance(name):
    file = "attendance.csv"
    if os.path.exists(file):
        df = pd.read_csv(file)
    else:
        df = pd.DataFrame(columns=["Name", "Time"])
    if name not in df["Name"].values:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df.loc[len(df)] = [name, now]
        df.to_csv(file, index=False)

# Detect face and return cropped face
def detect_face(frame):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104,117,123))
    face_detector.setInput(blob)
    detections = face_detector.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > 0.7:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            x1,y1,x2,y2 = box.astype("int")
            return frame[y1:y2, x1:x2]
    return None

# Generate 128-d embedding for a face
def get_embedding(face_img):
    face_blob = cv2.dnn.blobFromImage(face_img, 1.0/255, (96,96), (0,0,0), swapRB=True, crop=False)
    face_embedder.setInput(face_blob)
    vec = face_embedder.forward()
    return vec.flatten()

# Load known embeddings
def load_known_faces(folder):
    known_embeddings = []
    names = []
    for file in os.listdir(folder):
        if file.endswith(".jpg") or file.endswith(".png"):
            img = cv2.imread(os.path.join(folder, file))
            face = detect_face(img)
            if face is not None:
                embed = get_embedding(face)
                known_embeddings.append(embed)
                names.append(os.path.splitext(file)[0])
    return known_embeddings, names

known_embeddings, known_names = load_known_faces("known_faces")

# Webcam
cap = cv2.VideoCapture(0)
print("[INFO] Press 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    face = detect_face(frame)
    label = "Unknown"

    if face is not None:
        try:
            embed = get_embedding(face)
            similarities = cosine_similarity([embed], known_embeddings)[0]
            best_match_index = np.argmax(similarities)
            if similarities[best_match_index] > 0.5:
                label = known_names[best_match_index]
        except:
            pass

    mark_attendance(label)
    cv2.putText(frame, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.imshow("Face Attendance", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
