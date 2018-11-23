import numpy as np
import cv2
import matplotlib
import matplotlib.pyplot as plt
matplotlib.style.use("default")

# Let's import an Star Trek computer panel, now in color:
panel = cv2.imread("images/PC.png")


# Use the OpenCV native image navigator:
cv2.imshow('image', panel)
cv2.waitKey(0)
cv2.destroyAllWindows()


# Surprise when showing the image with matplotlib:
plt.figure()
plt.imshow(panel, interpolation="none")
plt.show()


# Deal with color spaces:
panel_rgb = cv2.cvtColor(panel, cv2.COLOR_BGR2RGB)

plt.figure()
plt.imshow(panel_rgb, interpolation="none")
plt.show()

# Let's see them decomposed:
plt.figure()
plt.subplot(241)
plt.imshow(panel, interpolation="none")
plt.subplot(245)
plt.imshow(panel_rgb, interpolation="none")
for i in range(3):
    plt.subplot(2, 4, i + 2)
    plt.imshow(panel[:, :, i], interpolation="none", cmap="gray")
    plt.xlabel("BRG"[i])
    plt.subplot(2, 4, i + 1 + 5)
    plt.imshow(panel_rgb[:, :, i], interpolation="none", cmap="gray")
    plt.xlabel("RGB"[i])
plt.show()

# Other color spaces:
plt.figure()
plt.subplot(241)
plt.imshow(panel_rgb, interpolation="none")
plt.subplot(245)
plt.imshow(panel_rgb, interpolation="none")
for i in range(3):
    plt.subplot(2, 4, i + 2)
    plt.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)[:, :, i], interpolation="none", cmap="gray")
    plt.xlabel("HSV"[i])
    plt.subplot(2, 4, i + 1 + 5)
    plt.imshow(cv2.cvtColor(panel, cv2.COLOR_BGR2LAB)[:, :, i], interpolation="none", cmap="gray")
    plt.xlabel("LAB"[i])
plt.show()


# Extract all the red buttons of the image:
panel_hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
lower_red = np.array([0, 150, 150])
upper_red = np.array([10, 255, 255])
mask = cv2.inRange(panel_hsv, lower_red, upper_red)
red_buttons = cv2.bitwise_and(panel_rgb, panel_rgb, mask=mask)

plt.figure()
plt.subplot(131)
plt.imshow(panel_rgb, interpolation="none")
plt.subplot(132)
plt.imshow(mask, interpolation="none", cmap="gray")
plt.subplot(133)
plt.imshow(red_buttons, interpolation="none")
plt.show()


# Clean the mask with morphological transformations:
panel_hsv = cv2.cvtColor(panel, cv2.COLOR_BGR2HSV)
lower_red = np.array([0, 150, 150])
upper_red = np.array([10, 255, 255])

mask = cv2.inRange(panel_hsv, lower_red, upper_red)
kernel = np.ones((3, 3), np.uint8)
mask = cv2.dilate(mask, kernel, iterations=2)

red_buttons = cv2.bitwise_and(panel_rgb, panel_rgb, mask=mask)

plt.figure()
plt.subplot(131)
plt.imshow(panel_rgb, interpolation="none")
plt.subplot(132)
plt.imshow(mask, interpolation="none", cmap="gray")
plt.subplot(133)
plt.imshow(red_buttons, interpolation="none")
plt.show()
