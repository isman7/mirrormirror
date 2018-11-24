import numpy as np
import cv2
import matplotlib.pyplot as plt

# Let's load a not-yet-Emperor terran Spook:
spook = cv2.imread("images/MindMeld.png", cv2.IMREAD_GRAYSCALE)

# Let's try to identify the corners of the image:
corners = cv2.cornerHarris(spook.astype(np.float32), blockSize=2, ksize=3, k=0.04)

corners = cv2.dilate(corners, None)
spook_corners = spook.copy()

spook_corners[corners > 0.01*corners.max()] = 255

plt.figure()
plt.subplot(121)
plt.imshow(spook, interpolation="none", cmap="gray")
plt.subplot(122)
plt.imshow(spook_corners, interpolation="none", cmap="gray")
plt.show()

# Sophisticated corner detection, SIFT:
sift = cv2.xfeatures2d.SIFT_create()
keypoints = sift.detect(spook, None)

spook_sift = cv2.drawKeypoints(spook.copy(), keypoints, None)

plt.figure()
plt.subplot(121)
plt.imshow(spook, interpolation="none", cmap="gray")
plt.subplot(122)
plt.imshow(spook_sift, interpolation="none")
plt.show()

# ORB:

orb = cv2.ORB_create()
keypoints = orb.detect(spook, None)

spook_orb = cv2.drawKeypoints(spook.copy(), keypoints, None)

plt.figure()
plt.subplot(121)
plt.imshow(spook, interpolation="none", cmap="gray")
plt.subplot(122)
plt.imshow(spook_orb, interpolation="none")
plt.show()