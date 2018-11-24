import numpy as np
import cv2
import matplotlib.pyplot as plt

# Let's find againg the buttons from the panel with template matching:
panel = cv2.imread("images/PC.png", cv2.IMREAD_GRAYSCALE)
button = cv2.imread("images/Button.png", cv2.IMREAD_GRAYSCALE)
w, h = button.shape

result = cv2.matchTemplate(panel, button, cv2.TM_CCOEFF_NORMED)

threshold = 0.8
loc = np.where( result >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(panel, pt, (pt[0] + w, pt[1] + h), 255, 2)

plt.figure()
plt.imshow(panel, interpolation='none', cmap='gray')
plt.show()

# Let's use S channel:
panel = cv2.imread("images/PC.png")
button = cv2.imread("images/Button.png")
w, h, *_ = button.shape

panel_S = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)[:, :, 1]
button_S = cv2.cvtColor(button, cv2.COLOR_BGR2HSV)[:, :, 1]

result = cv2.matchTemplate(panel_S, button_S, cv2.TM_CCOEFF_NORMED)

panel = cv2.cvtColor(panel, cv2.COLOR_BGR2GRAY)

threshold = 0.8
loc = np.where( result >= threshold)
for pt in zip(*loc[::-1]):
    cv2.rectangle(panel, pt, (pt[0] + w, pt[1] + h), 255, 2)

plt.figure()
plt.imshow(panel, interpolation='none', cmap='gray')
plt.show()


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

