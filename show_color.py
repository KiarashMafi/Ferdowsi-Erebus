import cv2
import numpy as np
import matplotlib.pyplot as plt

# Define the HSV color value
hsv = np.uint8([[[27, 232, 184]]])

# Convert the HSV color to BGR
bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

# Create a NumPy array of the BGR color
color = np.full((100, 100, 3), bgr[0][0])

# Display the color using Matplotlib
plt.imshow(color)
plt.show()