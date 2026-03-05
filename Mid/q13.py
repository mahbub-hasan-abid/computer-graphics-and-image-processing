import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1) Read image in grayscale
img = cv2.imread("image.png", cv2.IMREAD_GRAYSCALE)
if img is None:
    raise FileNotFoundError("image.png not found in the same folder")

# 2) Histogram Equalization (OpenCV built-in)
eq = cv2.equalizeHist(img)

# 3) Compute histograms (for showing)
hist_img, _ = np.histogram(img.flatten(), bins=256, range=[0, 256])
hist_eq, _  = np.histogram(eq.flatten(),  bins=256, range=[0, 256])

# 4) Show images + histograms
plt.figure(figsize=(12, 6))

plt.subplot(2, 2, 1)
plt.imshow(img, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 2, 2)
plt.imshow(eq, cmap="gray")
plt.title("Equalized Image")
plt.axis("off")

plt.subplot(2, 2, 3)
plt.plot(hist_img)
plt.title("Original Histogram")
plt.xlim([0, 255])

plt.subplot(2, 2, 4)
plt.plot(hist_eq)
plt.title("Equalized Histogram")
plt.xlim([0, 255])

plt.tight_layout()

# Save output (so VS Code te window na asleo output pabe)
cv2.imwrite("equalized_output.jpg", eq)
plt.savefig("equalization_result.png", dpi=200)

print("Saved: equalized_output.jpg and equalization_result.png")
plt.show(block=True)