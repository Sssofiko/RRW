import cv2
import numpy as np
import os

def load_image(image_path):
    """Загружает изображение и возвращает его в формате NumPy."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print("Ошибка: не удалось загрузить изображение.")
    return image

def load_watermark(watermark_path):
    """Загружает водяной знак как изображение в градациях серого."""
    watermark = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    if watermark is None:
        print("Ошибка: не удалось загрузить водяной знак.")
    return watermark

def convert_to_ycrcb(image):
    """Преобразует изображение в цветовое пространство YCrCb."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)

def apply_dct(y_channel):
    """Выполняет ДКП (DCT) на канале яркости изображения."""
    return cv2.dct(np.float32(y_channel))

def apply_idct(dct_coefficients):
    """Выполняет обратное ДКП (IDCT) для получения модифицированного изображения."""
    return cv2.idct(dct_coefficients)

def save_image(image, output_path):
    """Сохраняет изображение в формате JPEG с качеством 90%."""
    cv2.imwrite(output_path, image, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print(f"Файл сохранен: {output_path}")

def embed_watermark(image_path, watermark_path, output_path="watermarked_image.jpg"):
    """Встраивает водяной знак в изображение с использованием DCT."""

    image = load_image(image_path)
    if image is None:
        return

    watermark = load_watermark(watermark_path)
    if watermark is None:
        return

    if watermark.shape[0] > image.shape[0] or watermark.shape[1] > image.shape[1]:
        print("Ошибка: водяной знак больше, чем изображение.")
        return

    image_ycrcb = convert_to_ycrcb(image)
    y_channel = image_ycrcb[:, :, 0]

    dct_coeff = apply_dct(y_channel)
    dct_coeff[:watermark.shape[0], :watermark.shape[1]] += watermark

    y_channel_modified = apply_idct(dct_coeff)
    image_ycrcb[:, :, 0] = np.clip(y_channel_modified, 0, 255)

    watermarked_image = cv2.cvtColor(image_ycrcb, cv2.COLOR_YCrCb2BGR)
    save_image(watermarked_image, output_path)

def load_watermarked_image(image_path):
    """Загружает изображение с встроенным водяным знаком."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        print("Ошибка: не удалось загрузить изображение с водяным знаком.")
    return image

def extract_y_channel(image):
    """Извлекает яркостный канал Y из изображения в формате YCrCb."""
    image_ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
    return image_ycrcb[:, :, 0]

def compute_dct(y_channel):
    """Выполняет DCT на канале яркости для извлечения коэффициентов."""
    return cv2.dct(np.float32(y_channel))

def save_extracted_watermark(watermark, output_path):
    """Сохраняет извлеченный водяной знак в текстовый файл."""
    np.savetxt(output_path, watermark, fmt="%d")
    print(f"Извлеченный водяной знак сохранен в {output_path}")

def extract_watermark(watermarked_image_path, watermark_size, output_path="extracted_watermark.png"):
    """Извлекает водяной знак, усиливает контраст и устраняет шум."""

    image = load_watermarked_image(watermarked_image_path)
    if image is None:
        return

    y_channel = extract_y_channel(image)
    dct_coeff = compute_dct(y_channel)

    extracted_watermark = dct_coeff[:watermark_size[0], :watermark_size[1]]

    # Усиление контраста
    extracted_watermark -= np.mean(extracted_watermark)
    extracted_watermark = np.clip(extracted_watermark * 10, -128, 128)
    extracted_watermark -= extracted_watermark.min()
    extracted_watermark = (extracted_watermark / extracted_watermark.max()) * 255
    extracted_watermark = extracted_watermark.astype(np.uint8)

    # Фильтрация шума (Гауссово размытие)
    extracted_watermark = cv2.GaussianBlur(extracted_watermark, (3, 3), 0)

    # Бинаризация изображения
    _, extracted_watermark = cv2.threshold(extracted_watermark, 128, 255, cv2.THRESH_BINARY)

    cv2.imwrite(output_path, extracted_watermark)
    print(f"Извлеченный водяной знак сохранен в {output_path}")


def main():
    print("Выберите действие:")
    print("1 - Встраивание водяного знака")
    print("2 - Извлечение водяного знака")

    choice = input("Введите 1 или 2: ")

    if choice == "1":
        image_path = input("Введите путь к изображению: ")
        watermark_path = input("Введите путь к файлу с водяным знаком: ")
        output_path = input("Введите путь для сохранения результата (или нажмите Enter для watermarked_image.jpg): ")
        if not output_path:
            output_path = "watermarked_image.jpg"
        embed_watermark(image_path, watermark_path, output_path)

    elif choice == "2":
        watermarked_image_path = input("Введите путь к изображению с водяным знаком: ")
        watermark_height = int(input("Введите высоту водяного знака: "))
        watermark_width = int(input("Введите ширину водяного знака: "))
        output_path = input("Введите путь для сохранения результата (или нажмите Enter для extracted_watermark.txt): ")
        if not output_path:
            output_path = "extracted_watermark.txt"
        extract_watermark(watermarked_image_path, (watermark_height, watermark_width), output_path)

    else:
        print("Ошибка: неверный ввод. Введите 1 или 2.")

if __name__ == "__main__":
    main()
