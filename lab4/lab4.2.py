import dlib
from skimage import io
from scipy.spatial import distance
import requests
from bs4 import BeautifulSoup

sp = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
facerec = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
detector = dlib.get_frontal_face_detector()

url = 'https://acts.kpi.ua/uk/peoples/'

page_data = requests.get(url).text

images_urls = [
    image.attrs.get('src')
    for image in BeautifulSoup(page_data, 'lxml').find_all('img')
]

images_urls = [
    image_url
    for image_url in images_urls
    if image_url and len(image_url) > 0
]

print(images_urls)
