import cv2
import numpy as np
import matplotlib.pyplot as plt

from Constructing_robust_features import (
    extract_frequency_matrices,
    divide_into_cells,
    compute_difference_statistics,
    compute_robust_features
)
from Shifting_the_histogram_of_robust_features import (
    merge_cells,
    replace_frequency_matrices_in_blocks,
    compute_shifted_cells
)
from Review_of_JPEG_compression import (
    split_into_blocks, merge_blocks,
    apply_dct, apply_idct,
    quantize, dequantize,
    JPEG_QUANT_MATRIX
)


def embed_watermark(cover_path=None, watermark_path=None,
                    block_size=8, selected_bands=None, input_bands=None,
                    cell_height=None, cell_width=None, key=None):
    """
    Встраивает водяной знак, полученный из watermark.jpg, в изображение cover_path.
    Параметр selected_bands – список индексов (по зигзагу) DCT-коэффициентов, в которые будет встраивание.
    Например, для R=1: selected_bands=[11], для R=3: selected_bands=[10,11,12].

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

    # Извлечение коэффициентов выбранных полос (например, [11] или [10,11,12])
    freq_matrices = extract_frequency_matrices(quantized_blocks, selected_bands, M, N)
    cells = divide_into_cells(freq_matrices, cell_height, cell_width)
    R = len(selected_bands)  # число выбранных частотных полос

    # Используем переданный ключ для вычисления разностных статистик и робастных признаков
    eta_before = compute_difference_statistics(cells, key)
    robust_features_before = compute_robust_features(eta_before)

    # Чтение водяного изображения и получение битовой последовательности
    watermark_img = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    if watermark_img is None:
        raise FileNotFoundError(f"Не удалось открыть файл {watermark_path}")
    wmk_bits = (watermark_img.flatten() >= 128).astype(np.int32)
    L = cells.shape[1]  # число ячеек (один бит на ячейку)
    if len(wmk_bits) < L:
        watermark_bits = np.concatenate([wmk_bits, np.zeros(L - len(wmk_bits), dtype=np.int32)])
    elif len(wmk_bits) > L:
        watermark_bits = wmk_bits[:L]
    else:
        watermark_bits = wmk_bits

    # Фиксируем глобальную полярность: первый бит равен 1
    watermark_bits[0] = 1
    # Определение порога T: выбираем T так, чтобы T > max(|λ|) для обратимости
    T = np.max(np.abs(robust_features_before)) + 1
    print(T)
    # recommended_T = np.max(np.abs(robust_features_before)) + 1
    # print(f"\nРекомендуемый порог T: {recommended_T}")
    #
    # # Предложение пользователю задать свой порог T или использовать рекомендуемый
    # T_input = input(f"Желаете задать собственное значение T (рекомендуемое: {recommended_T}) или использовать рекомендованное? (1 - собственное, 2 - рекомендуемое): ").strip().lower()
    #
    # if T_input == '1':
    #     try:
    #         T = float(input("Введите собственное значение порога T: ").strip())
    #     except ValueError:
    #         print("❌ Неверный формат ввода, используется рекомендованное значение.")
    #         T = recommended_T
    # elif T_input == '2':
    #     T = recommended_T
    # else:
    #     print("\n❌ Некорректный выбор. Пожалуйста, выберите 1 или 2.")

    # Встраивание: симметричный сдвиг гистограмм робастных признаков
    wm_cells = compute_shifted_cells(cells, watermark_bits, T, R)
    wm_freq_matrices = merge_cells(wm_cells, cell_height, cell_width, M, N)
    wm_quantized_blocks = replace_frequency_matrices_in_blocks(
        quantized_blocks, wm_freq_matrices, selected_bands, M, N
    )

    # Обратное квантование, IDCT и сборка итогового изображения
    wm_dct_blocks = dequantize(wm_quantized_blocks, JPEG_QUANT_MATRIX)
    wm_spatial_blocks = apply_idct(wm_dct_blocks)
    watermarked_img = merge_blocks(wm_spatial_blocks, cover_img.shape, block_size=block_size)

    # Предложение показать гистограммы
    show_histograms = input(
        "\nЖелаете посмотреть гистограммы robust features до и после встраивания? (Да/Нет): ").strip().lower()
    if show_histograms == "да":
        show_robust_features_histograms(cover_path=cover_path,
                                        watermark_path=watermark_path,
                                        block_size=8,
                                        selected_bands=input_bands,
                                        cell_height=cell_height,
                                        cell_width=cell_width,
                                        key=key)

    return watermarked_img, watermark_bits, T, (h, w)


def correct_global_polarity(bits, expected_first=1):
    """
    Проверяет глобальную полярность битовой последовательности.
    Если первый бит не равен expected_first, инвертирует всю последовательность.
    """
    if bits[0] != expected_first:
        bits = 1 - bits
    return bits


def extract_watermark(watermarked_img, cover_shape, block_size=8,
                      selected_bands=None, cell_height=8, cell_width=4, key=None):
    """
    Извлекает битовую последовательность из watermarked_img, используя те же параметры,
    что и при встраивании. Параметр selected_bands – список частотных полос.
    Если глобальная полярность неверна (первый бит извлечён равен 0), инвертирует последовательность.
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

    # Извлекаем коэффициенты выбранных полос и делим на ячейки
    freq_matrices = extract_frequency_matrices(quantized_blocks, selected_bands, M, N)
    cells = divide_into_cells(freq_matrices, cell_height, cell_width)

    # Вычисляем робастные признаки
    eta = compute_difference_statistics(cells, key)
    robust_features = compute_robust_features(eta)

    # Извлечение: если λ(k) >= 0, то бит = 1, иначе 0
    extracted_bits = (robust_features >= 0).astype(int)
    extracted_bits = correct_global_polarity(extracted_bits)

    return extracted_bits


def show_robust_features_histograms(cover_path=None, watermark_path=None,
                                    block_size=8, selected_bands=None,
                                    cell_height=None, cell_width=None, key=None):
    """
    Показывает гистограммы робастных признаков до и после встраивания водяного знака.

    Параметры:
      robust_features_before: np.array - массив робастных признаков до встраивания.
      robust_features_after: np.array - массив робастных признаков после встраивания.
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

    # Извлечение коэффициентов выбранных полос (например, [11] или [10,11,12])
    freq_matrices = extract_frequency_matrices(quantized_blocks, selected_bands, M, N)
    cells = divide_into_cells(freq_matrices, cell_height, cell_width)
    R = len(selected_bands)  # число выбранных частотных полос

    # Используем переданный ключ для вычисления разностных статистик и робастных признаков
    eta_before = compute_difference_statistics(cells, key)
    robust_features_before = compute_robust_features(eta_before)

    # Чтение водяного изображения и получение битовой последовательности
    watermark_img = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    if watermark_img is None:
        raise FileNotFoundError(f"Не удалось открыть файл {watermark_path}")
    wmk_bits = (watermark_img.flatten() >= 128).astype(np.int32)
    L = cells.shape[1]  # число ячеек (один бит на ячейку)
    if len(wmk_bits) < L:
        watermark_bits = np.concatenate([wmk_bits, np.zeros(L - len(wmk_bits), dtype=np.int32)])
    elif len(wmk_bits) > L:
        watermark_bits = wmk_bits[:L]
    else:
        watermark_bits = wmk_bits

    # Фиксируем глобальную полярность: первый бит равен 1
    watermark_bits[0] = 1
    # Определение порога T: выбираем T так, чтобы T > max(|λ|) для обратимости
    T = np.max(np.abs(robust_features_before)) + 1

    # Встраивание: симметричный сдвиг гистограмм робастных признаков
    wm_cells = compute_shifted_cells(cells, watermark_bits, T, R)
    wm_freq_matrices = merge_cells(wm_cells, cell_height, cell_width, M, N)
    wm_quantized_blocks = replace_frequency_matrices_in_blocks(
        quantized_blocks, wm_freq_matrices, selected_bands, M, N
    )

    # Обратное квантование, IDCT и сборка итогового изображения
    wm_dct_blocks = dequantize(wm_quantized_blocks, JPEG_QUANT_MATRIX)
    wm_spatial_blocks = apply_idct(wm_dct_blocks)
    watermarked_img = merge_blocks(wm_spatial_blocks, cover_img.shape, block_size=block_size)

    # Дополнительно можно построить гистограммы робастных признаков до и после
    freq_matrices_after = extract_frequency_matrices(wm_quantized_blocks, selected_bands, M, N)
    cells_after = divide_into_cells(freq_matrices_after, cell_height, cell_width)
    eta_after = compute_difference_statistics(cells_after, key)
    robust_features_after = compute_robust_features(eta_after)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.hist(robust_features_before, bins=30, color='orange')
    plt.title("Histogram before watermarking")
    plt.xlabel("Value")
    plt.ylabel("Count")

    plt.subplot(1, 2, 2)
    neg_vals = robust_features_after[robust_features_after < 0] - T
    pos_vals = robust_features_after[robust_features_after >= 0] + T
    plt.hist(neg_vals, bins=30, color='red', alpha=0.5, label='bit-0-region')
    plt.hist(pos_vals, bins=30, color='blue', alpha=0.5, label='bit-1-region')
    plt.title("Histogram after watermarking")
    plt.xlabel("Value")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.show()
