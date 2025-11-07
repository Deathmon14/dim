
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os


IMG_PATH = "image4.tif"   # any grayscale/colour image file


img = cv2.imread(IMG_PATH)
if img is None:
    raise FileNotFoundError(f"Could not read: {IMG_PATH}")

img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


noise_sigma = 25
noise = np.random.normal(0, noise_sigma, img_gray.shape).astype(np.int16)
noisy = np.clip(img_gray.astype(np.int16) + noise, 0, 255).astype(np.uint8)

os.makedirs("images", exist_ok=True)
cv2.imwrite("images/noisy_image.jpg", noisy)
print("Saved noisy image -> images/noisy_image.jpg")


gauss = cv2.GaussianBlur(noisy, (5,5), 0)      # Gaussian blur
median = cv2.medianBlur(noisy, 5)              # Median filter


dft = cv2.dft(np.float32(noisy), flags=cv2.DFT_COMPLEX_OUTPUT)
dft_shift = np.fft.fftshift(dft)


rows, cols = noisy.shape
crow, ccol = rows//2, cols//2
radius = 30  # try 20–50 in viva
mask = np.zeros((rows, cols, 2), np.uint8)
cv2.circle(mask, (ccol, crow), radius, (1,1), thickness=-1)

fshift = dft_shift * mask
f_ishift = np.fft.ifftshift(fshift)
img_back = cv2.idft(f_ishift)
# magnitude to real image
freq_filtered = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])
freq_filtered = cv2.normalize(freq_filtered, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


plt.figure(figsize=(12,8))
plt.subplot(2,3,1); plt.imshow(img_gray, cmap='gray'); plt.title("Original (Gray)"); plt.axis('off')
plt.subplot(2,3,2); plt.imshow(noisy,    cmap='gray'); plt.title("Noisy (σ=25)");   plt.axis('off')
plt.subplot(2,3,3); plt.imshow(gauss,    cmap='gray'); plt.title("Gaussian 5×5");   plt.axis('off')
plt.subplot(2,3,4); plt.imshow(median,   cmap='gray'); plt.title("Median 5×5");     plt.axis('off')
plt.subplot(2,3,5); plt.imshow(freq_filtered, cmap='gray'); plt.title("Freq LPF (r=30)"); plt.axis('off')
plt.tight_layout(); plt.show()


cv2.imwrite("images/denoise_gaussian.jpg", gauss)
cv2.imwrite("images/denoise_median.jpg", median)
cv2.imwrite("images/denoise_frequency.jpg", freq_filtered)
print("Saved: images/denoise_gaussian.jpg, images/denoise_median.jpg, images/denoise_frequency.jpg")
