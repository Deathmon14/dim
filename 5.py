# prog5_segmentation_edges_threshold.py
# pip install opencv-python matplotlib

import cv2, matplotlib.pyplot as plt
IMG = "image5.tif"
img = cv2.imread(IMG, cv2.IMREAD_GRAYSCALE)

edges = cv2.Canny(img, 100, 200)

_, th_global = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
th_adapt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                 cv2.THRESH_BINARY, 11, 2)

plt.figure(figsize=(12,8))
plt.subplot(2,2,1); plt.imshow(img, cmap='gray');      plt.title("Input"); plt.axis('off')
plt.subplot(2,2,2); plt.imshow(edges, cmap='gray');    plt.title("Canny")
plt.subplot(2,2,3); plt.imshow(th_global, cmap='gray');plt.title("Global Th=127"); plt.axis('off')
plt.subplot(2,2,4); plt.imshow(th_adapt, cmap='gray'); plt.title("Adaptive (Gaussian)"); plt.axis('off')
plt.tight_layout(); plt.show()
