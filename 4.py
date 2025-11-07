
import cv2, numpy as np, matplotlib.pyplot as plt

img = cv2.imread("image4.tif", cv2.IMREAD_GRAYSCALE)


noise = np.random.normal(0, 25, img.shape)
noisy = np.clip(img + noise, 0, 255).astype(np.uint8)


gauss  = cv2.GaussianBlur(noisy, (5,5), 0)
median = cv2.medianBlur(noisy, 5)


dft = cv2.dft(np.float32(noisy), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)
rows, cols = noisy.shape
crow, ccol = rows//2, cols//2
mask = np.zeros((rows, cols, 2), np.uint8)
cv2.circle(mask, (ccol, crow), 30, (1,1), -1)
fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
freq = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
freq = cv2.normalize(freq, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


titles = ["Original", "Noisy", "Gaussian", "Median", "Freq LPF"]
imgs   = [img, noisy, gauss, median, freq]
for i in range(5):
    plt.subplot(2,3,i+1); plt.imshow(imgs[i], cmap='gray'); plt.title(titles[i]); plt.axis('off')
plt.tight_layout(); plt.show()
