
import cv2, matplotlib.pyplot as plt

IMG = "images/image2.tif"
img = cv2.imread(IMG, cv2.IMREAD_GRAYSCALE)

hist = cv2.calcHist([img],[0],None,[256],[0,256])
eq   = cv2.equalizeHist(img)
hist_eq = cv2.calcHist([eq],[0],None,[256],[0,256])

plt.figure(figsize=(10,6))
plt.subplot(2,2,1); plt.imshow(img, cmap='gray'); plt.title("Original"); plt.axis('off')
plt.subplot(2,2,2); plt.plot(hist);    plt.title("Hist Original")
plt.subplot(2,2,3); plt.imshow(eq,  cmap='gray'); plt.title("Equalized"); plt.axis('off')
plt.subplot(2,2,4); plt.plot(hist_eq); plt.title("Hist Equalized")
plt.tight_layout(); plt.show()
