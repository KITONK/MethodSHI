from cv2 import imread
from cv2 import imshow
import time
from cv2 import waitKey
from cv2 import destroyAllWindows
from cv2 import CascadeClassifier
from cv2 import rectangle
from cv2 import cvtColor
from cv2 import COLOR_BGR2GRAY
import numpy
import cv2
import urllib
import numpy as np

# №1

# pixels = imread('dance.jpg')
face_cascade = CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
smile_cascade = CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
eye_cascade = CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
nose_cascade = CascadeClassifier('haarcascade_nose.xml')

# gray_filter = cvtColor(pixels, COLOR_BGR2GRAY)
# faces = face_cascade.detectMultiScale(pixels, 1.3, 7)
#
# for (x, y, w, h) in faces:
#     rectangle(pixels, (x, y), (x+w, y+h), (255, 0, 0), 1)
#     roi_gray = gray_filter[y:y+h, x:x+w]
#     roi_color = pixels[y:y+h, x:x+w]
#     smile = smile_cascade.detectMultiScale(roi_gray, 1.1, 1)
#     eye = eye_cascade.detectMultiScale(roi_gray, 1.01, 3)
#     nose = nose_cascade.detectMultiScale(roi_color, 1.1, 3)
#     for (ex, ey, ew, eh) in smile:
#         rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 0, 255), 1)
#     for(sx, sy, sw, sh) in eye:
#         rectangle(roi_color, (sx, sy), (sx+sw, sy+sh), (0, 255, 0), 1)
#     for(mx, my, mw, mh) in nose:
#          rectangle(roi_color, (mx, my), (mx+mw, my+mh), (255, 255, 0), 1)
#
# print(f'Found {len(faces)} faces!')
# imshow('face detection', pixels)
# waitKey(0)
# destroyAllWindows()

# Video
# №2

# cap = cv2.VideoCapture(0)
#
#
# while(True):
#     time.sleep(0.02)
#     ret, pixels = cap.read()
#
#     gray = cv2.cvtColor(pixels, cv2.COLOR_BGR2GRAY)
#     faces = face_cascade.detectMultiScale(
#         gray,
#         scaleFactor=1.3,
#         minNeighbors=5
#     )
#
#     for (x, y, w, h) in faces:
#         rectangle(pixels, (x, y), (x + w, y + h), (255, 0, 0), 2)
#         roi_gray = gray[y:y + h, x:x + w]
#         roi_color = pixels[y:y + h, x:x + w]
#         smile = smile_cascade.detectMultiScale(roi_gray, 1.2, 20)
#         eye = eye_cascade.detectMultiScale(roi_gray, 1.1, 20)
#
#         for (ex, ey, ew, eh) in smile:
#             rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 0, 255), 1)
#         for (sx, sy, sw, sh) in eye:
#             rectangle(roi_color, (sx, sy), (sx + sw, sy + sh), (0, 255, 0), 1)
#
#         cv2.imshow('video', pixels)
#
#
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
#
# cap.release()
# cv2.destroyAllWindows()


# НАХОЖДЕНИЕ ЛЮДЕЙ НА ВИДЕО
# ИСПОЛЬЗОВАНИЕ HOG-классификаторов
# №3

from imutils.object_detection import non_max_suppression
import imutils

video_path = 'video-with-people.mp4'
cap = cv2.VideoCapture(video_path)
#
# fourcc = cv2.VideoWriter_fourcc(*'mp4v')
# out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (800, 450))
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())


def detect_body(image):
    (rects, weights) = hog.detectMultiScale(image, winStride=(4, 4), padding=(8, 8), scale=1.05)
    rects = np.array([[x, y, x + w, y + h] for (x, y, w, h) in rects])
    pick = non_max_suppression(rects, probs=None, overlapThresh=0.65)
    for (xA, yA, xB, yB) in pick:
        if xB > 570:
            continue
        cv2.rectangle(image, (xA, yA), (xB, yB), (0, 255, 0), 2)

def process_and_save():
    # cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1280, 720))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = imutils.resize(frame, width=800, height=450)
            detect_body(frame)
            out.write(frame)
        else:
            break
    out.release()

def procces_and_show():
    for i in range(150):
        cap.read()
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            frame = imutils.resize(frame, width=800, height=450)
            detect_body(frame)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break


procces_and_show()

cap.release()
cv2.destroyAllWindows()

