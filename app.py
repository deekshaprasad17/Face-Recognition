from flask import Flask, render_template, Response
import cv2
import numpy as np
import face_recognition
import os
import pandas as pd
from datetime import datetime
import dlib
from scipy.spatial import distance as dist

app = Flask(__name__)

# ---- Blink Detection Setup ----
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    return (A + B) / (2.0 * C)

EYE_AR_THRESH = 0.23
EYE_AR_CONSEC_FRAMES = 3
blink_counter = 0
blinked = False
recognized_names = set()

# ---- Face Recognition Setup ----
path = 'ImagesAttendance'
images, classNames = [], []
for cl in os.listdir(path):
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])
print("Loaded images:", classNames)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodeList.append(face_recognition.face_encodings(img)[0])
    return encodeList

encodeListKnown = findEncodings(images)
print("Encoding complete ✅")

# ---- Dlib for landmark detection ----
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
(lStart, lEnd), (rStart, rEnd) = (42, 48), (36, 42)

# ---- Attendance Handler ----
def markAttendance(name):
    now = datetime.now()
    dtString = now.strftime('%Y-%m-%d,%H:%M:%S')
    with open('attendance.csv', 'a+') as f:
        f.seek(0)
        lines = f.readlines()
        names_today = [line.split(',')[0] for line in lines if dtString.split(',')[0] in line]
        if name not in names_today:
            f.write(f"{name},{dtString}\n")

# ---- Video Generator ----
cap = cv2.VideoCapture(0)

def generate_frames():
    global blink_counter, blinked
    while True:
        success, img = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        rects = detector(gray)
        for rect in rects:
            shape = predictor(gray, rect)
            shape_np = np.array([[p.x, p.y] for p in shape.parts()])
            leftEye = shape_np[lStart:lEnd]
            rightEye = shape_np[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            if ear < EYE_AR_THRESH:
                blink_counter += 1
            else:
                if blink_counter >= EYE_AR_CONSEC_FRAMES:
                    blinked = True
                blink_counter = 0

        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)
        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)
            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = [v * 4 for v in faceLoc]
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                if blinked and name not in recognized_names:
                    cv2.putText(img, f"{name} ✅", (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 1, (255,255,255), 2)
                    markAttendance(name)
                    recognized_names.add(name)
                    blinked = False
                else:
                    cv2.putText(img, f"{name} (Blink to mark)", (x1+6, y2-6), cv2.FONT_HERSHEY_COMPLEX, 0.7, (0,0,255), 2)

        # Convert to JPEG and yield
        ret, buffer = cv2.imencode('.jpg', img)
        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video')
def video():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/attendance')
def attendance():
    df = pd.read_csv('attendance.csv', names=['Name', 'Date', 'Time'])
    data = df.tail(20).to_dict(orient='records')
    return render_template('index.html', rows=data)

if __name__ == "__main__":
    app.run(debug=True)
