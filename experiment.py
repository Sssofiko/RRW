import numpy as np
import os
import matplotlib.pyplot as plt
from Embedding_extraction import embed_watermark
from quality import psnr, ssim
from Review_of_JPEG_compression import jpeg_compression_pipeline


def get_key_from_file():
    """
    Функция для загрузки ключа из файла 'key.txt'.
    """
    if os.path.isfile("key.txt"):
        with open("key.txt", "r") as f:
            key = list(map(int, f.read().split()))
            return np.array(key)
    else:
        raise FileNotFoundError("Файл 'key.txt' не найден.")


def perform_experiment(image_paths, watermark_path, input_bands):
    """
    Функция для выполнения эксперимента с встраиванием водяного знака и вычислением PSNR и SSIM.

    Параметры:
    - image_paths: список путей к изображениям для эксперимента.
    - watermark_path: путь к водяному знаку.
    - selected_bands: список выбранных полос для встраивания.
    """
    key = get_key_from_file()  # Загружаем ключ из файла
    selected_bands = [64 - band for band in input_bands]
    psnr_values = []
    ssim_values = []
    watermarked_images = []

    for image_path in image_paths:
        print(f"Обрабатываем изображение: {image_path}")

        # Встраиваем водяной знак
        watermarked_img, watermark_bits, T, cover_shape = embed_watermark(
            cover_path=image_path,
            watermark_path=watermark_path,
            block_size=8,
            selected_bands=selected_bands,
            cell_height=8,
            cell_width=4,
            key=key  # Передаем ключ
        )

        # Загружаем исходное изображение
        original_img = jpeg_compression_pipeline(image_path)

        # Рассчитываем метрики PSNR и SSIM
        psnr_value = psnr(original_img, watermarked_img)
        ssim_value = ssim(original_img, watermarked_img)

        print(f"PSNR для изображения {image_path}: {psnr_value:.2f} dB")
        print(f"SSIM для изображения {image_path}: {ssim_value:.4f}")

        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        watermarked_images.append(watermarked_img)

    return psnr_values, ssim_values, watermarked_images

def main():
    # Пути к изображениям и водяному знаку
    image_paths = ["Lena.jpg", "Boats.jpg", "Airplane.jpg"]
    watermark_path = "bk.jpg"

    # Варианты выбранных полос
    selected_bands_list = [
        [11],  # Один индекс полосы
        [11, 15],  # Два индекса полос
        [11, 15, 20]  # Три индекса полос
    ]

    # Настройка вывода для каждого изображения
    fig, axes = plt.subplots(3, 4, figsize=(12, 10))

    # Вставка изображений в соответствии с выбранными полосами
    for i, selected_bands in enumerate(selected_bands_list):
        print(f"\n--- Эксперимент с выбранными полосами: {selected_bands} ---")
        psnr_values, ssim_values, watermarked_images = perform_experiment(image_paths, watermark_path, selected_bands)

        for j, image_path in enumerate(image_paths):
            original_img = jpeg_compression_pipeline(image_path)
            # Отображение изображений в сетке
            axes[j, i].imshow(original_img, cmap='gray')
            axes[j, i].axis('off')

            if i == 0:  # Лейблы для столбцов
                axes[j, i].set_title(f"(a) {image_path.split('.')[0]}", fontsize=10)

            # Отображение изображений с водяным знаком
            watermarked_img = watermarked_images[j]
            axes[j, i + 1].imshow(watermarked_img, cmap='gray')
            axes[j, i + 1].axis('off')

            if i == 0 or i == 1 or i == 2:  # Лейблы для строк
                axes[j, i + 1].set_title(f"PSNR: {psnr_values[j]:.2f} dB\nSSIM: {ssim_values[j]:.4f}", fontsize=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
