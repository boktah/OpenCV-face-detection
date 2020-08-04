import cv2
import pickle

# Note colors are in BGR
blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)

face_cascade_alt_2 = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml")

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels = {"person-name": 1}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

cap = cv2.VideoCapture(0)


while True:
    success, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    roi = frame
    roi_gray = gray

    faces = face_cascade_alt_2.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3, minSize=(60, 60))

    for x, y, w, h in faces:
        frame = cv2.rectangle(frame, (x, y), (x + w, y + h), blue, 5)
        roi = frame[y:y+h, x:x+w]
        roi_gray = gray[y:y+h, x:x+w]

        id_, conf = recognizer.predict(roi_gray)

        font = cv2.FONT_HERSHEY_SIMPLEX
        conf = str(conf)
        conf = (conf[:5]) if len(conf) > 5 else conf
        name = labels[id_] + ",  " + conf
        color = (0, 255, 0)
        stroke = 2

        cv2.putText(frame, name, (x,y), font, 0.8, color, stroke, cv2.LINE_AA)

    cv2.imshow("Webcam Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

