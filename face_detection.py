import cv2

blue = (255, 0, 0)
green = (0, 255, 0)
red = (0, 0, 255)

face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")
face_cascade_alt = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt.xml")
face_cascade_alt_2 = cv2.CascadeClassifier("cascades/haarcascade_frontalface_alt2.xml")
profile_cascade = cv2.CascadeClassifier("cascades/haarcascade_profileface.xml")
eye_cascade = cv2.CascadeClassifier("cascades/haarcascade_eye.xml")
smile_cascade = cv2.CascadeClassifier("cascades/haarcascade_smile.xml")

cap = cv2.VideoCapture(0)


def draw_rects(input_cascade, color):
    global img, grey_img

    faces = input_cascade.detectMultiScale(grey_img, scaleFactor=1.1, minNeighbors=3)

    for x, y, w, h in faces:
        img = cv2.rectangle(img, (x, y), (x + w, y + h), color, 5)

    # rect_img = cv2.resize(rect_img,(int(rect_img.shape[1] / 4), int(rect_img.shape[0] / 4)))

    return img


while True:
    success, img = cap.read()
    grey_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # img = draw_rects(face_cascade, red)
    # img = draw_rects(face_cascade_alt, green)
    # img = draw_rects(face_cascade_alt_2, blue)
    # img = draw_rects(profile_cascade, blue)
    # img = draw_rects(eye_cascade, blue)
    # img = draw_rects(smile_cascade, blue)

    cv2.imshow("Webcam Face Detection", img)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

