from sklearn.neighbors import KNeighborsClassifier
import cv2
import pickle
import numpy as np
import os
import csv
from datetime import datetime
import time

# Initialize video capture
cap = cv2.VideoCapture(0)

# Set desired frame size
desired_width = 640
desired_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, desired_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, desired_height)

# Retrieve and print the actual frame size
actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
print(f"Frame size set to: {actual_width}x{actual_height}")

facedetect = cv2.CascadeClassifier('C:\\Users\\Antu Sanbui\\Desktop\\haarcascades\\haarcascade_frontalcatface.xml')

with open('C:\\Users\\Antu Sanbui\\Desktop\\haarcascades\\names.pkl' , 'rb') as f:
    LABELS = pickle.load(f)
with open('C:\\Users\\Antu Sanbui\\Desktop\\haarcascades\\face_data.pkl', 'rb') as f:
    FACES = pickle.load(f)

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(FACES, LABELS)

COL_NAMES = ['NAMES', 'TIME']

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY)
    faces = facedetect.detectMultiScale(gray, 1.3, 5)
    for(x,y,w,h) in faces:
        padding = 0.2
        x_new = int(x - w * padding / 2)
        y_new = int(y - h * padding / 2)
        w_new = int(w * (1 + padding))
        h_new = int(h * (1 + padding))
        crop_img = frame[y_new:y_new+h_new, x_new:x_new+w_new, :]
        resized_img = cv2.resize(crop_img,(50,50)).flatten().reshape(1,-1)
        output = knn.predict(resized_img)
        t = time.time()
        date = datetime.fromtimestamp(t).strftime("%d-%m-%Y")
        timestamp = datetime.fromtimestamp(t).strftime("%H-%M-%S")
        cv2.putText(frame, str(output[0]), (x,y-15), cv2.FONT_HERSHEY_COMPLEX, 1, (32,255,100), 1)
        cv2.rectangle(frame, (x_new, y_new), (x_new+w_new, y_new+h_new), (0,255,255),1)
        attendence = [str(output[0]), str(timestamp)]
        exist = os.path.isfile("C:\\Users\\Antu Sanbui\\Desktop\project\\Automated_Attendence_System\\attendence" + date + ".csv")
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('o'):
        if exist:
            with open("C:\\Users\\Antu Sanbui\\Desktop\project\\Automated_Attendence_System\\attendence" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(attendence)
            csvfile.close()
        else:
            with open("C:\\Users\\Antu Sanbui\\Desktop\project\\Automated_Attendence_System\\attendence" + date + ".csv", "+a") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(COL_NAMES)
                writer.writerow(attendence)
            csvfile.close()
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
