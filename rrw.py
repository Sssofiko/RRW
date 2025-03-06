#############################################
# embedding_extraction.py
#############################################

import cv2
import numpy as np
import matplotlib.pyplot as plt

# Импорт функций из ранее созданных файлов:
from Constructing_robust_features import (
    extract_frequency_matrices,
    divide_into_cells,
    generate_random_bijection,
    compute_difference_statistics,
    compute_robust_features
)
from Shifting_the_histogram_of_robust_features import (
    merge_cells,
    replace_frequency_matrices_in_blocks
)
from Review_of_JPEG_compression import (
    split_into_blocks, merge_blocks,
    apply_dct, apply_idct,
    quantize, dequantize,
    JPEG_QUANT_MATRIX
)

#############################################
# Симметричный сдвиг: упрощённая версия
#############################################

def compute_shifted_cell_symmetric(cell, w, T):
    """
    Для данной ячейки вычисляет исходное значение робастного признака:
      lambda_orig = sum_{i=0}^{m*n-1} { x[i] * sign(i) }
    Затем, чтобы сдвинуть robust feature ровно на +T (если w==1)
    или на -T (если w==0), просто прибавляем нужное значение к первому элементу.
    """
    m, n = cell.shape
    total = m * n
    signs = np.array([1 if i % 2 == 0 else -1 for i in range(total)])
    flat = cell.flatten()
    lambda_orig = np.sum(flat * signs)
    d = T if w == 1 else -T  # желаемое изменение
    flat[0] += d
    return flat.reshape(m, n)

def compute_shifted_cells_symmetric(cells, w_bits, T, R):
    """
    Применяет симметричный сдвиг для всех ячеек во всех выбранных частотных полосах,
    используя compute_shifted_cell_symmetric.
    """
    R_val, L, m, n = cells.shape
    shifted_cells = np.empty_like(cells)
    for r in range(R_val):
        for k in range(L):
            shifted_cells[r, k] = compute_shifted_cell_symmetric(cells[r, k], w_bits[k], T)
    return shifted_cells

#############################################
# Функции встраивания и извлечения водяного знака
#############################################

def embed_watermark(cover_path="Lena.jpg", watermark_path="watermark.jpg",
                    block_size=8, selected_band=11, cell_height=8, cell_width=4):
    """
    Встраивает водяной знак, полученный из watermark.jpg, в изображение cover_path.
    Параметры: R=1, sigma={11}, ячейки 8x4.

    Возвращает:
      watermarked_img (np.array) – итоговое водяное изображение,
      watermark_bits (np.array) – исходная битовая последовательность (длина L),
      key – приватный ключ (биекция) для перестановки в ячейках,
      T – порог, использованный при встраивании,
      cover_shape – размер исходного изображения.
    """
    # Загрузка исходного изображения
    cover_img = cv2.imread(cover_path, cv2.IMREAD_GRAYSCALE)
    if cover_img is None:
        raise FileNotFoundError(f"Не удалось открыть файл {cover_path}")
    h, w = cover_img.shape

    # Разбиение на 8x8 блоки, DCT и квантование
    blocks = split_into_blocks(cover_img, block_size=block_size)
    dct_blocks = apply_dct(blocks)
    quantized_blocks = quantize(dct_blocks, JPEG_QUANT_MATRIX)
    M = h // block_size
    N = w // block_size

    # Извлечение коэффициентов выбранной полосы (sigma = {11})
    freq_matrices = extract_frequency_matrices(quantized_blocks, [selected_band], M, N)
    cells = divide_into_cells(freq_matrices, cell_height, cell_width)

    # Генерация ключа и вычисление робастных признаков до встраивания
    key = generate_random_bijection(cell_height, cell_width)
    eta_before = compute_difference_statistics(cells, key)
    robust_features_before = compute_robust_features(eta_before)

    # Чтение водяного изображения и получение битовой последовательности
    watermark_img = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    if watermark_img is None:
        raise FileNotFoundError(f"Не удалось открыть файл {watermark_path}")
    wmk_bits = (watermark_img.flatten() >= 128).astype(np.int32)
    L = cells.shape[1]  # число ячеек, требуемая длина водяного знака
    if len(wmk_bits) < L:
        watermark_bits = np.concatenate([wmk_bits, np.zeros(L - len(wmk_bits), dtype=np.int32)])
    elif len(wmk_bits) > L:
        watermark_bits = wmk_bits[:L]
    else:
        watermark_bits = wmk_bits

    # Фиксируем глобальную полярность: задаем первый бит равным 1
    watermark_bits[0] = 1
    # Определение порога T
    T = np.max(np.abs(robust_features_before)) + 1

    # Встраивание: симметричный сдвиг гистограмм робастных признаков
    wm_cells = compute_shifted_cells_symmetric(cells, watermark_bits, T, R=1)
    wm_freq_matrices = merge_cells(wm_cells, cell_height, cell_width, M, N)
    wm_quantized_blocks = replace_frequency_matrices_in_blocks(
        quantized_blocks, wm_freq_matrices, [selected_band], M, N
    )

    # Обратное квантование, IDCT и сборка итогового изображения
    wm_dct_blocks = dequantize(wm_quantized_blocks, JPEG_QUANT_MATRIX)
    wm_spatial_blocks = apply_idct(wm_dct_blocks)
    watermarked_img = merge_blocks(wm_spatial_blocks, cover_img.shape, block_size=block_size)

    # Повторное вычисление робастных признаков после встраивания для построения гистограмм
    freq_matrices_after = extract_frequency_matrices(wm_quantized_blocks, [selected_band], M, N)
    cells_after = divide_into_cells(freq_matrices_after, cell_height, cell_width)
    eta_after = compute_difference_statistics(cells_after, key)
    robust_features_after = compute_robust_features(eta_after)

    # Построение гистограмм
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(robust_features_before, bins=30, color='orange')
    plt.title("Histogram before watermarking")
    plt.xlabel("Value")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    neg_vals = robust_features_after[robust_features_after < 0]
    pos_vals = robust_features_after[robust_features_after >= 0]
    plt.hist(neg_vals, bins=30, color='red', alpha=0.5, label='bit-0-region')
    plt.hist(pos_vals, bins=30, color='blue', alpha=0.5, label='bit-1-region')
    plt.title("Histogram after watermarking")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()

    return watermarked_img, watermark_bits, key, T, (h, w)

def correct_global_polarity(bits, expected_first=1):
    """
    Проверяет глобальную полярность битовой последовательности.
    Если первый бит не равен expected_first (по умолчанию 1), инвертирует всю последовательность.

    Параметры:
      bits: np.array – извлечённая битовая последовательность (0 или 1)
      expected_first: ожидаемое значение первого бита (по умолчанию 1)

    Возвращает:
      np.array – скорректированная битовая последовательность.
    """
    if bits[0] != expected_first:
        bits = 1 - bits
    return bits

def extract_watermark(watermarked_img, cover_shape, block_size=8,
                      selected_band=11, cell_height=8, cell_width=4, key=None):
    """
    Извлекает битовую последовательность из watermarked_img, используя те же параметры,
    что и при встраивании (R=1, sigma={11}, ячейки 8×4).

    Если глобальная полярность неверна (первый бит извлечён равен 0),
    инвертирует всю последовательность.

    Возвращает:
      extracted_bits: np.array – извлечённая битовая последовательность.
    """
    if key is None:
        raise ValueError("Не задан key, невозможно корректно извлечь биты.")

    h, w = cover_shape
    wm_img = np.array(watermarked_img, dtype=np.uint8)

    # Разбиение, DCT, квантование
    blocks = split_into_blocks(wm_img, block_size=block_size)
    dct_blocks = apply_dct(blocks)
    quantized_blocks = quantize(dct_blocks, JPEG_QUANT_MATRIX)
    M = h // block_size
    N = w // block_size

    # Извлекаем выбранную частотную полосу и делим на ячейки
    freq_matrices = extract_frequency_matrices(quantized_blocks, [selected_band], M, N)
    cells = divide_into_cells(freq_matrices, cell_height, cell_width)

    # Вычисляем робастные признаки
    eta = compute_difference_statistics(cells, key)
    robust_features = compute_robust_features(eta)

    # Извлечение: если λ(k) >= 0, то бит = 1, иначе 0

    extracted_bits = (robust_features >= 0).astype(int)
    extracted_bits = correct_global_polarity(extracted_bits)

    return extracted_bits


if __name__ == "__main__":
    # 1) Встраивание водяного знака из watermark.jpg в Lena.jpg
    wm_img, original_bits, key, T, cover_shape = embed_watermark(
        cover_path="Lena.jpg",
        watermark_path="BK.jpg",
        block_size=8,
        selected_band=11,
        cell_height=8,
        cell_width=4
    )
    cv2.imwrite("Lena_watermarked.jpg", wm_img)
    print("Watermarked image saved as Lena_watermarked.jpg")
    print(f"Исходная битовая последовательность (длина): {len(original_bits)}")

    # 2) Извлечение водяного знака
    extracted_bits = extract_watermark(
        watermarked_img=wm_img,
        cover_shape=cover_shape,
        block_size=8,
        selected_band=11,
        cell_height=8,
        cell_width=4,
        key=key
    )
    print(f"Извлечённая битовая последовательность (длина): {len(extracted_bits)}")
    print("Исходные биты:")
    print(original_bits)
    print("Извлечённые биты:")
    print(extracted_bits)
    num_errors = np.sum(original_bits != extracted_bits)
    ber = num_errors / len(original_bits) * 100
    print(f"Ошибок: {num_errors} из {len(original_bits)}  => BER = {ber:.2f}%")
    if num_errors == 0:
        print("Водяной знак извлечён без ошибок!")
    else:
        print("Есть расхождения между исходными и извлечёнными битами.")

    # 3) Сохранение извлечённого водяного знака как бинарного изображения.
    # Предполагаем, что длина водяного знака равна 128 бит, что соответствует размеру 8x16 пикселей.
    extracted_watermark_img = (extracted_bits.reshape((8, 16)) * 255).astype(np.uint8)
    cv2.imwrite("extracted_watermark.jpg", extracted_watermark_img)
    print("Извлечённый водяной знак сохранён как extracted_watermark.jpg")
