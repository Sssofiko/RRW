from Review_of_JPEG_compression import jpeg_compression_pipeline
import matplotlib.pyplot as plt
import cv2
from quality import psnr
original_img = jpeg_compression_pipeline("Lena.jpg")
img = jpeg_compression_pipeline("1.jpg")

plt.figure(figsize=(12, 6))

# Исходное изображение
plt.subplot(1, 2, 1)
plt.imshow(original_img, cmap='gray')
plt.title("Исходное изображение")
plt.axis('off')

# Изображение с водяным знаком
plt.subplot(1, 2, 2)
plt.imshow(img, cmap='gray')
plt.title("Изображение с водяным знаком")
plt.axis('off')

# Показываем сравнение
plt.tight_layout()
plt.show()


img1 = cv2.imread("Lena.jpg", cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread("фвшднф.jpg", cv2.IMREAD_GRAYSCALE)
img3 = cv2.imread("ффф.jpg", cv2.IMREAD_GRAYSCALE)
print(psnr(img3,img2))
