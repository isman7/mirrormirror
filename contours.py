import numpy as np
import cv2
import matplotlib.pyplot as plt

# Let's import an Star Trek computer panel:
panel = cv2.imread("images/PC.png", cv2.IMREAD_GRAYSCALE)

# To look for contours, we need to threshold the image:
ret, panel_simple = cv2.threshold(panel, 127, 255, cv2.THRESH_BINARY)
panel_adaptive_mean = cv2.adaptiveThreshold(panel, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
                                            blockSize=11, C=2)
panel_adaptive_gauss = cv2.adaptiveThreshold(panel, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                             blockSize=11, C=2)

plt.figure()
plt.subplot(141)
plt.imshow(panel, interpolation="none", cmap="gray")
plt.subplot(142)
plt.imshow(panel_simple, interpolation="none", cmap="gray")
plt.xlabel("Simple threshold")
plt.subplot(143)
plt.imshow(panel_adaptive_mean, interpolation="none", cmap="gray")
plt.xlabel("Mean threshold")
plt.subplot(144)
plt.imshow(panel_adaptive_gauss, interpolation="none", cmap="gray")
plt.xlabel("Gaussian threshold")
plt.show()

# Let's find the contours
panel_contours, contours, hierarchy = cv2.findContours(panel_adaptive_mean.copy(),
                                                       cv2.RETR_TREE,
                                                       cv2.CHAIN_APPROX_SIMPLE)

plt.figure()
plt.subplot(131)
plt.imshow(panel, interpolation="none", cmap="gray")
plt.subplot(132)
plt.imshow(panel_adaptive_mean, interpolation="none", cmap="gray")
plt.xlabel("Mean threshold")
plt.subplot(133)
plt.imshow(panel_contours, interpolation="none", cmap="gray")
plt.xlabel("Contours image")
plt.show()

# Obtain all the buttons by their area:
filtered_contours = [cnt for cnt in contours if 200 < cv2.contourArea(cnt) < 600]
mask = np.zeros(panel.shape, dtype=np.uint8)
cv2.drawContours(mask, filtered_contours, -1, 255, -1)

kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

panel_rgb = cv2.cvtColor(cv2.imread("images/PC.png"), cv2.COLOR_BGR2RGB)
buttons = cv2.bitwise_and(panel_rgb, panel_rgb, mask=mask)

plt.figure()
plt.subplot(131)
plt.imshow(panel_rgb, interpolation="none")
plt.subplot(132)
plt.imshow(mask, interpolation="none", cmap="gray")
plt.subplot(133)
plt.imshow(buttons, interpolation="none")
plt.show()


# Clean the mask with aspect ratio of the contours:
def check_aspect(x, y, w, h):
    aspect_ratio = w/h
    return 0.5 < aspect_ratio < 2


filtered_contours = [cnt for cnt in filtered_contours if check_aspect(*cv2.boundingRect(cnt))]
mask = np.zeros(panel.shape, dtype=np.uint8)
cv2.drawContours(mask, filtered_contours, -1, 255, -1)

kernel = np.ones((5, 5), np.uint8)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

panel_rgb = cv2.cvtColor(cv2.imread("images/PC.png"), cv2.COLOR_BGR2RGB)
buttons = cv2.bitwise_and(panel_rgb, panel_rgb, mask=mask)

plt.figure()
plt.subplot(131)
plt.imshow(panel_rgb, interpolation="none")
plt.subplot(132)
plt.imshow(mask, interpolation="none", cmap="gray")
plt.subplot(133)
plt.imshow(buttons, interpolation="none")
plt.show()

