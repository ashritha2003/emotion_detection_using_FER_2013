
import cv2
import os
import numpy as np
from keras.models import model_from_json
import matplotlib.pyplot as plt
l = []

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# load json and create model
json_file = open('model/emotion_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
emotion_model = model_from_json(loaded_model_json)

# load weights into new model
emotion_model.load_weights("model/emotion_model.h5")
print("Loaded model from disk")

# start the webcam feed
cap = cv2.VideoCapture(0)

# pass here your video path
# you may download one from here : https://www.pexels.com/video/three-girls-laughing-5273028/
#cap = cv2.VideoCapture("C:\\Users\\hp\\OneDrive\\Desktop\\ed 1\\emotions.mp4")

while True:
    # Find haar cascade to draw bounding box around face
    ret, frame = cap.read()
    frame = cv2.resize(frame, (1280, 720))
    if not ret:
        break
    face_detector = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # detect faces available on camera
    num_faces = face_detector.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    # take each face available on the camera and Preprocess it
    for (x, y, w, h) in num_faces:
        cf=0
        cv2.rectangle(frame, (x, y-50), (x+w, y+h+10), (0, 255, 0), 4)
        roi_gray_frame = gray_frame[y:y + h, x:x + w]
        cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray_frame, (48, 48)), -1), 0)

        # predict the emotions
        emotion_prediction = emotion_model.predict(cropped_img)
        maxindex = int(np.argmax(emotion_prediction))
        l.append(emotion_dict[maxindex])
        k = set(l)
        m = str(len(k))
        cv2.putText(frame, emotion_dict[maxindex], (x+5, y-20), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame,"No of expressions changed:",(x-300,y+450),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, m, (x + 150, y + 450), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2,
                    cv2.LINE_AA)
    keys = [0,0,0,0,0,0,0]

    ha = l.count("Happy")
    percent = (ha / (len(l)+1)) * 100
    keys[0] = percent
    print("Happy-" + str(int(percent)) + "%")
    Aa = l.count("Angry")
    percent6 = (Aa / (len(l)+1)) * 100
    keys[1]= percent6
    print("Angry-" + str(int(percent6)) + "%")
    Ne = l.count("Neutral")
    percent1 = (Ne / (len(l)+1)) * 100
    keys[2] = percent1
    print("Neutral-" + str(int(percent1)) + "%")
    Sa = l.count("Sad")
    percent2 = (Sa / (len(l)+1)) * 100
    keys[3] = percent2
    print("Sad-" + str(int(percent2)) + "%")
    Fe = l.count("Fearful")
    percent3 = (Fe / (len(l)+1))* 100
    keys[4] = percent3
    print("Fearful-" + str(int(percent3)) + "%")
    Su = l.count("Surprised")
    percent4 = (Su / (len(l)+1)) * 100
    keys[5] = percent4
    print("Surprised-" + str(int(percent4)) + "%")
    Di = l.count("Disgusted")
    percent5 = (Di / (len(l)+1)) * 100
    keys[6] = percent5
    print("Disgusted-" + str(int(percent5)) + "%")
    values = ['Happy', 'Angry', 'Neutral', 'Sad', 'Fearful', 'Surprised', 'Disgusted']
    print(keys)
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#plt.bar(values, keys)
#plt.show()

cap.release()
cv2.destroyAllWindows()
