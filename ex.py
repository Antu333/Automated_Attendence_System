import cv2
import pickle
import numpy as np
import os

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

face_data = []
i = 0
name = input("Enter your name: ")

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
        resized_img = cv2.resize(crop_img,(50,50))
        if len(face_data)<=100 and i%10 == 0:
            face_data.append(resized_img)
        #i = i+1
        cv2.putText(frame, str(len(face_data)),(50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (36,255,12))
        cv2.rectangle(frame, (x_new, y_new), (x_new+w_new, y_new+h_new), (0,255,255),1)
    
    cv2.imshow("Frame", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q') or len(face_data) == 100:
        break

cap.release()
cv2.destroyAllWindows()

face_data = np.asarray(face_data)
face_data = face_data.reshape(100, -1)

if 'names.pickle' not in os.listdir('C:\\Users\\Antu Sanbui\\Desktop\\haarcascades'):
    names = [name]*100
    with open('C:\\Users\\Antu Sanbui\\Desktop\\haarcascades\\names.pkl' , 'wb') as f:
        pickle.dump(names,f)
else:
    with open('C:\\Users\\Antu Sanbui\\Desktop\\haarcascades\\names.pkl' , 'rb') as f:
        names = pickle.load(f)
    names = names+[names]*100
    with open('C:\\Users\\Antu Sanbui\\Desktop\\haarcascades\\names.pkl' , 'wb') as f:
        pickle.dump(names,f)

if 'face_data.pickle' not in os.listdir('C:\\Users\\Antu Sanbui\\Desktop\\haarcascades'):
    with open('C:\\Users\\Antu Sanbui\\Desktop\\haarcascades\\face_data.pkl' , 'wb') as f:
        pickle.dump(face_data,f)
else:
    with open('C:\\Users\\Antu Sanbui\\Desktop\\haarcascades\\face_data.pkl', 'rb') as f:
        faces = pickle.load(f)
    facess = np.append(faces, face_data, axis=0)
    with open('C:\\Users\\Antu Sanbui\\Desktop\\haarcascades\\face_data.pkl' , 'wb') as f:
        pickle.dump(names,f)