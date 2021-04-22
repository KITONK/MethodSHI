# import cv2
# import dlib
# from scipy.spatial import distance
# from skimage import io
#
# sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
# detector = dlib.get_frontal_face_detector()
#
# # ---------- ЧАСТЬ 1 ---------- #
# cap = cv2.VideoCapture(0)
# for i in range(30):
#     cap.read()
#
# ret, frame = cap.read()
# cv2.imwrite('cam.jpg', frame)
# cap.release()
# cv2.destroyAllWindows()
#
# img = io.imread('2.jpg')
# img1 = io.imread('cam.jpg')
# win1 = dlib.image_window()
# win1.clear_overlay()
# win1.set_image(img)
# win1.set_image(img1)
#
# dets = detector(img, 1)
# dets1 = detector(img1, 1)
# for k, d in enumerate(dets):
#     print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
#         k, d.left(), d.top(), d.right(), d.bottom()))  # малюємо рамку навколо знайденого обличчя
#     shape = sp(img, d)
#     win1.clear_overlay()
#     # win1.add_overlay(d)
#     # win1.add_overlay(shape)
#     # win1.wait_until_closed()
#
#     face_descriptor1 = facerec.compute_face_descriptor(img, shape)
#
# for k, d in enumerate(dets1):
#     print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
#         k, d.left(), d.top(), d.right(), d.bottom()))  # малюємо рамку навколо знайденого обличчя
#     shape1 = sp(img1, d)
#     win1.clear_overlay()
#     win1.add_overlay(d)
#     win1.add_overlay(shape1)
#     win1.wait_until_closed()
#
#     face_descriptor2 = facerec.compute_face_descriptor(img1, shape1)
#
# a = distance.euclidean(face_descriptor1, face_descriptor2)
# print(a)

import dlib
import cv2
from scipy.spatial import distance


# creating models' objects
sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

# proccessing the first picture
# image_1 = cv2.imread('elon_before.jpg')
cap = cv2.VideoCapture(0)
for i in range(30):
    cap.read()

ret, frame = cap.read()
cv2.imwrite('cam.jpg', frame)
cap.release()
cv2.destroyAllWindows()

image_1 = cv2.imread('cam.jpg')
win1 = dlib.image_window()
win1.clear_overlay()
win1.set_image(image_1)

faces_1 = detector(image_1, 1)
for k, d in enumerate(faces_1):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))  # малюємо рамку навколо знайденого обличчя
    shape = sp(image_1, d)
    win1.clear_overlay()
    win1.add_overlay(d)
    win1.add_overlay(shape)
    win1.wait_until_closed()

    face_descriptor_1 = facerec.compute_face_descriptor(image_1, shape)

# proccessing the second picture
image_2 = cv2.imread('2.jpg')
faces_2 = detector(image_2, 1)
for k, d in enumerate(faces_2):
    print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
        k, d.left(), d.top(), d.right(), d.bottom()))  # малюємо рамку навколо знайденого обличчя
    shape = sp(image_2, d)
    win1.clear_overlay()
    win1.add_overlay(d)
    win1.add_overlay(shape)
    win1.wait_until_closed()

    face_descriptor_2 = facerec.compute_face_descriptor(image_2, shape)

# calculating euclidean distance
result = distance.euclidean(face_descriptor_1, face_descriptor_2)

if result < 0.6:
    print('Most likely there is one person')
else:
    print('Most likely there are two different people')