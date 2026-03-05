import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read image in grayscale
img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)

# Check if image loaded
if img is None:
    raise FileNotFoundError("Image not found")

# Number of bins
B = 32

# Compute histogram
hist, bins = np.histogram(img.flatten(), bins=B, range=[0,256])

# Plot histogram
plt.figure(figsize=(6,4))
plt.bar(bins[:-1], hist, width=8, color='gray')
plt.title("Histogram with 32 Bins")
plt.xlabel("Pixel Intensity")
plt.ylabel("Frequency")
plt.show()