import numpy as np
import cv2
import matplotlib.pyplot as plt

# Let's import faces and eyes classifiers from OpenCV:
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Kirk's image:
kirk = cv2.imread("images/WTFKirk0.png", cv2.IMREAD_GRAYSCALE)

# Face detection routine:
faces = face_cascade.detectMultiScale(kirk, 1.3, 5)
for x, y, w, h in faces:
    cv2.rectangle(kirk, (x,y),(x+w,y+h), 255,2)

plt.figure()
plt.imshow(kirk, interpolation='none', cmap='gray')
plt.show()


# Trying it over different images:
def detect_face(img):
    img_return = img.copy()
    faces = face_cascade.detectMultiScale(img_return, 1.3, 5)
    for x, y, w, h in faces:
        cv2.rectangle(img_return, (x,y),(x+w,y+h), 255,2)
    return img_return


kirk = cv2.imread("images/WTFKirk0.png", cv2.IMREAD_GRAYSCALE)
spook = cv2.imread("images/ColdBloodedLogic.png", cv2.IMREAD_GRAYSCALE)
uhura = cv2.imread("images/KinkyUhura.png", cv2.IMREAD_GRAYSCALE)

plt.figure()
plt.subplot(131)
plt.imshow(detect_face(kirk), interpolation='none', cmap='gray')
plt.subplot(132)
plt.imshow(detect_face(spook), interpolation='none', cmap='gray')
plt.subplot(133)
plt.imshow(detect_face(uhura), interpolation='none', cmap='gray')
plt.show()

