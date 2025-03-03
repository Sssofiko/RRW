import cv2
import numpy as np

# Импортируем необходимые функции из наших модулей.
# Предполагается, что файлы Review_of_JPEG_compression.py,
# Constructing_robust_features.py и RRW_framework_based_on_histogram_shifting.py находятся в той же директории.
from Review_of_JPEG_compression import split_into_blocks, apply_dct, quantize, dequantize, apply_idct, merge_blocks, JPEG_QUANT_MATRIX, zigzag_scan, inverse_zigzag_scan
from Constructing_robust_features import generate_random_bijection
from RRW_framework_based_on_histogram_shifting import simulate_RRW_framework

def compute_zigzag_indices(n=8):
    """
    Возвращает список координат (i, j) для матрицы n×n согласно зигзагообразному обходу.
    """
    indices = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            # для четных s идем сверху вниз
            for i in range(min(s, n - 1), max(-1, s - n), -1):
                j = s - i
                indices.append((i, j))
        else:
            # для нечетных s идем снизу вверх
            for i in range(max(0, s - n + 1), min(s, n - 1) + 1):
                j = s - i
                indices.append((i, j))
    return indices

def main():
    # Пути к входным изображениям (cover и watermark)
    cover_path = "Lena1.jpg"       # путь к cover-изображению (JPEG)
    watermark_path = "watermark.jpg"  # путь к водяному знаку (JPEG)

    # Загружаем cover-изображение в градациях серого
    cover_img = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
    if cover_img is None:
        print("Cover image not found!")
        return
    h, w = cover_img.shape
    print(f"Cover image loaded: {h} x {w}")

    # Разбиваем изображение на 8x8 блоки
    blocks = split_into_blocks(cover_img, block_size=8)
    print(f"Number of blocks: {blocks.shape[0]}")

    # Применяем DCT к каждому блоку
    dct_blocks = apply_dct(blocks)

    # Квантование блоков с использованием стандартной матрицы JPEG
    quantized_blocks = quantize(dct_blocks, JPEG_QUANT_MATRIX)

    # Для встраивания водяного знака нам нужно выбрать определённые коэффициенты (частотные полосы).
    # В статье для R = 1 рекомендуется использовать {σ1} = {11}.
    # Вычисляем зигзагообразное упорядочение координат для блока 8x8:
    zigzag_coords = compute_zigzag_indices(n=8)
    # Выбираем 11-ю координату (так как нумерация начинается с 0, используем индекс 10)
    selected_bands = [zigzag_coords[10]]
    print(f"Selected band (coordinate): {selected_bands[0]}")

    # Генерируем приватный ключ (перестановку) для ячеек.
    # Здесь размер ключа равен m*n. Пусть размер ячейки соответствует размеру блока (8x8) или может быть другим,
    # в зависимости от реализации; для простоты возьмём m=8, n=4 (как в примерах).
    m, n = 8, 4
    key_s = generate_random_bijection(m, n)
    print("Generated private key (permutation):", key_s)

    # Далее используем функцию simulate_RRW_framework, которая:
    # 1) Вычисляет надежные признаки до встраивания
    # 2) Выбирает порог T (T > max(|λ|))
    # 3) Загружает водяной знак, преобразует его в бинарную последовательность
    # 4) Встраивает водяной знак путём сдвига гистограммы надежных признаков
    # 5) Возвращает обновленные квантованные блоки и дополнительные данные.
    result = simulate_RRW_framework(quantized_blocks, cover_img.shape, watermark_path, selected_bands, key_s)
    T = result["T"]
    print("Selected threshold T:", T)

    updated_quantized_blocks = result["updated_quantized_blocks"]

    # Восстанавливаем watermarked JPEG-изображение:
    # 1. Деквантование
    dequantized_blocks = dequantize(updated_quantized_blocks, JPEG_QUANT_MATRIX)
    # 2. Применяем обратное DCT
    reconstructed_blocks = apply_idct(dequantized_blocks)
    # 3. Собираем блоки в итоговое изображение
    watermarked_img = merge_blocks(reconstructed_blocks, cover_img.shape, block_size=8)

    # Сохраняем полученное watermarked-изображение
    cv2.imwrite("watermarked.jpg", watermarked_img)
    print("Watermarked image saved as 'watermarked.jpg'.")

if __name__ == "__main__":
    main()
