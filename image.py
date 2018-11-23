import imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Let's import Kirk's WTF face with OpenCV
kirk = cv2.imread("images/WTFKirk0.png", cv2.IMREAD_GRAYSCALE)


# Use the OpenCV native image navigator:
cv2.imshow('image', kirk)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Images are numpy arrays:
assert isinstance(kirk, np.ndarray)

cv2.imshow('image', kirk[:300, :300])
cv2.waitKey(0)
cv2.destroyAllWindows()

# Saving the grayscale cropped image:
cv2.imwrite("images/WTFKirk0_grayscale.png", kirk[:300, :300])

# If images are numpy arrays... all matplotlib stack is working!
plt.figure()
plt.imshow(kirk, interpolation="none", cmap="gray")
plt.show()