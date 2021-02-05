from cv2 import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np

profileface_cascade = cv.CascadeClassifier('haarcascade_profileface.xml')
frontalface_cascade = cv.CascadeClassifier(
    'haarcascade_frontalface_default.xml')


def detect_and_blur_faces(img):
    face_img = img.copy()
    frontalface_rects = frontalface_cascade.detectMultiScale(
        face_img, scaleFactor=1.3, minNeighbors=5)
    profileface_rects = profileface_cascade.detectMultiScale(
        face_img, scaleFactor=1.3, minNeighbors=6)

    for(x, y, w, h) in frontalface_rects:
        blur_face = cv.medianBlur(face_img[y:y+h, x:x+w], 51)
        face_img[y:y+h, x:x+w] = blur_face
        cv.rectangle(face_img, (x, y), (x+w, y+h), (255, 0, 0), 3)

    for(x, y, w, h) in profileface_rects:
        blur_face = cv.medianBlur(face_img[y:y+h, x:x+w], 51)
        face_img[y:y+h, x:x+w] = blur_face
        cv.rectangle(face_img, (x, y), (x+w, y+h), (255, 0, 0), 3)

    return face_img


cap = cv.VideoCapture(0)

while True:
    _, frame = cap.read()
    faces = detect_and_blur_faces(frame)
    faces = cv.flip(faces, 1)
    cv.imshow('Webcam', faces)
    if (cv.waitKey(1) == ord('q')):
        break

cap.release()
cv.destroyAllWindows()
cv.imwrite("print.png", faces)
