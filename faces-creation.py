import cv2
from datetime import datetime
import os

cap = cv2.VideoCapture(0)

images = []

name = input("Enter subject name: ").lower().replace(" ", "-")

if not os.path.exists("resources/" + name):
    os.mkdir("resources/" + name)

while True:
    success, frame = cap.read()
    cv2.imshow("Webcam Face Detection", frame)

    key = cv2.waitKey(1)

    if key & 0xFF == ord("q"):
        break
    elif key & 0xFF == ord("a"):
        now = datetime.now()
        dt_string = now.strftime("%d_%m_%Y-%H_%M_%S")
        img_name = "resources/" + name + "/" + dt_string + ".png"
        images.append((img_name, frame))
        print("image taken")

for img in images:
    cv2.imwrite(img[0], img[1])
