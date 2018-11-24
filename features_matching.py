import cv2
import matplotlib.pyplot as plt

# Let's load a surprised Kirk:
kirk0 = cv2.imread("images/WTFKirk0.png", cv2.IMREAD_GRAYSCALE)
kirk1 = cv2.imread("images/WTFKirk1.png", cv2.IMREAD_GRAYSCALE)

plt.figure()
plt.subplot(121)
plt.imshow(kirk0, interpolation='none', cmap='gray')
plt.subplot(122)
plt.imshow(kirk1, interpolation='none', cmap='gray')
plt.show()

# Create the SIFT object, detect keypoints and compute descriptors:
sift = cv2.xfeatures2d.SIFT_create()
keypoints0, descriptors0 = sift.detectAndCompute(kirk0, None)
keypoints1, descriptors1 = sift.detectAndCompute(kirk1, None)

# Create a BF matcher to identify similar features in two different images:
bf = cv2.BFMatcher()
matches = bf.knnMatch(descriptors0, descriptors1, k=2)

good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append([m])


kirk_matched = cv2.drawMatchesKnn(kirk0, keypoints0, kirk1, keypoints1, good, None, flags=2)

plt.figure()
plt.imshow(kirk_matched, interpolation='none')
plt.show()

