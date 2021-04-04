import cv2
import imutils
import numpy as np
from matplotlib import pyplot as plt
from IPython.display import display, Markdown

video_path = 'doroga.mp4'
username = 'Kyrylo Shchupii IT-81'


def process(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blured_image = cv2.GaussianBlur(gray_image, (11, 11), 2)
    canny_image = cv2.Canny(blured_image, 50, 100)
    lines = cv2.HoughLinesP(canny_image, 1, np.pi / 180, 150, None, 50, 10)
    lines = [] if lines is None else lines
    for (x1, y1, x2, y2) in map(lambda x: x[0], lines):
        # 1280x720
        if y1 < 450 or y2 < 450:
            continue

        # пропустить горизонтальные линии
        if x1 != x2 and abs((y2 - y1) / (x2 - x1)) < 0.2:
            continue

        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 5, cv2.LINE_AA)
    cv2.putText(image, username, (50, 100), cv2.FONT_HERSHEY_COMPLEX, 3, (255, 255, 255), 3, cv2.LINE_4)


def process_and_show():
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            process(frame)
            cv2.imshow('Frame', frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()


def process_and_save():
    cap = cv2.VideoCapture(video_path)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output.mp4', fourcc, 20.0, (1280, 720))
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:
            process(frame)
            out.write(frame)
        else:
            break
    cap.release()
    out.release()
    cv2.destroyAllWindows()


process_and_save()
