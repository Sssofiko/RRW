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
    compute_difference_statistics,
    compute_robust_features
)
from Shifting_the_histogram_of_robust_features import (
    merge_cells,
    replace_frequency_matrices_in_blocks,
    compute_t_r,
    compute_shifted_cell,
    compute_shifted_cells
)
from Review_of_JPEG_compression import (
    split_into_blocks, merge_blocks
)


#############################################
# Симметричный сдвиг: упрощённая версия
#############################################

def compute_shifted_cell_symmetric(cell, w, T):
    """
    Для данной ячейки (размер m x n) выполняется симметричный сдвиг коэффициентов:
      Если рассматривать элементы с индексами ε = 1,..., m*n, то:
      для нечётных ε: x̃[ε] = x[ε] + (2*w-1) * floor((T + ε - 1) / (m*n))
      для чётных  ε: x̃[ε] = x[ε] - (2*w-1) * floor((T + ε - 1) / (m*n))

    Для R=1, t_1 = T.
    """
    # m, n = cell.shape
    # total = m * n
    # # Приводим ячейку к вектору (row-major)
    # flat = cell.flatten().astype(np.int32)
    # # Индексы ε от 1 до total (т.е. np.arange(total)+1)
    # eps = np.arange(total)+1
    # # Вычисляем сдвиги для каждого элемента
    # shifts = np.floor((T + eps - 1) / total).astype(int)
    # # Фактор: +1, если w==1, -1, если w==0
    # factor = 2 * w - 1
    # # Для нечётных индексов (eps нечётное) прибавляем, для чётных – вычитаем
    # odd_mask = (eps % 2 == 1)
    # flat_shifted = np.empty_like(flat)
    # flat_shifted[odd_mask] = flat[odd_mask] + factor * shifts[odd_mask]
    # flat_shifted[~odd_mask] = flat[~odd_mask] - factor * shifts[~odd_mask]
    #
    # return flat_shifted.reshape(m, n)

    # m, n = cell.shape
    # total = m * n
    # # Приводим ячейку к вектору (row-major)
    # flat = cell.flatten().astype(np.int32)
    # # Индексы ε от 1 до total (т.е. np.arange(total)+1)
    # eps = np.arange(total)
    # # Вычисляем сдвиги для каждого элемента
    # shifts = np.floor((T + eps - 1) / total).astype(int)
    # # Фактор: +1, если w==1, -1, если w==0
    # factor = 2 * w - 1
    # # Для нечётных индексов (eps нечётное) прибавляем, для чётных – вычитаем
    # odd_mask = (eps % 2 == 1)
    # flat_shifted = np.empty_like(flat)
    # flat_shifted[odd_mask] = flat[odd_mask] + factor * shifts[odd_mask]
    # flat_shifted[~odd_mask] = flat[~odd_mask] - factor * shifts[~odd_mask]
    # # Вычисляем робастный признак до и после сдвига.
    # # Здесь s[i] = +1, если i (0-индекс) чётное, и -1, если нечётное.
    # signs = np.array([1 if i % 2 == 0 else -1 for i in range(total)])
    # original_feature = np.sum(flat * signs)
    # shifted_feature = np.sum(flat_shifted * signs)
    # delta = shifted_feature - original_feature
    # # Желаемое изменение: +T если w==1, -T если w==0.
    # desired = T if w == 1 else -T
    # diff = desired - delta
    #
    # # Корректируем один элемент так, чтобы итоговый робастный признак изменился ровно на diff.
    # # При корректировке элемента с индексом i изменение робастного признака составит: adjustment * signs[i].
    # # Чтобы компенсировать diff, выберем i такое, чтобы скорректировать с минимальным искажением.
    # # Например, выбираем элемент, у которого уже произведённое изменение (|flat_shifted - flat|) минимально.
    # modification = np.abs(flat_shifted - flat)
    # idx = np.argmin(modification)
    # # Корректировка: мы хотим, чтобы signs[idx] * (adjustment) = diff, т.е.
    # adjustment = diff * (1 if signs[idx] > 0 else -1)
    # flat_shifted[idx] += adjustment
    #
    # # (Опционально можно проверить, что итоговое изменение точно соответствует)
    # new_feature = np.sum(flat_shifted * signs)
    # # Если new_feature != original_feature + desired, можно добавить небольшое округление
    # # assert new_feature == original_feature + desired, "Ошибка: робастный признак не соответствует целевому изменению"

    # return flat_shifted.reshape(m, n)

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
    Параметр R здесь задаёт число выбранных полос.
    """
    t = compute_t_r(T, R)
    R_val, L, m, n = cells.shape  # R_val должно совпадать с len(selected_bands)
    shifted_cells = np.empty_like(cells)
    for r in range(R_val):
        for k in range(L):
            shifted_cells[r, k] = compute_shifted_cell_symmetric(cells[r, k], w_bits[k], t[r])
    return shifted_cells


#############################################
# Функции встраивания и извлечения водяного знака
#############################################

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
    # dct_blocks = apply_dct(blocks)
    # quantized_blocks = quantize(dct_blocks, JPEG_QUANT_MATRIX)
    M = h // block_size
    N = w // block_size

    # Извлечение коэффициентов выбранных полос (например, [11] или [10,11,12])
    freq_matrices = extract_frequency_matrices(blocks, selected_bands, M, N)
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

    # # Фиксируем глобальную полярность: первый бит равен 1
    # watermark_bits[0] = 1
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
        blocks, wm_freq_matrices, selected_bands, M, N
    )

    # Обратное квантование, IDCT и сборка итогового изображения
    # wm_dct_blocks = dequantize(wm_quantized_blocks, JPEG_QUANT_MATRIX)
    # wm_spatial_blocks = apply_idct(wm_dct_blocks)
    watermarked_img = merge_blocks(wm_quantized_blocks, cover_img.shape, block_size=block_size)

    # Дополнительно можно построить гистограммы робастных признаков до и после
    # freq_matrices_after = extract_frequency_matrices(wm_quantized_blocks, selected_bands, M, N)
    # cells_after = divide_into_cells(freq_matrices_after, cell_height, cell_width)
    # eta_after = compute_difference_statistics(cells_after, key)
    # robust_features_after = compute_robust_features(eta_after)

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
    # dct_blocks = apply_dct(blocks)
    # quantized_blocks = quantize(dct_blocks, JPEG_QUANT_MATRIX)
    M = h // block_size
    N = w // block_size

    # Извлекаем коэффициенты выбранных полос и делим на ячейки
    freq_matrices = extract_frequency_matrices(blocks, selected_bands, M, N)
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
    # dct_blocks = apply_dct(blocks)
    # quantized_blocks = quantize(dct_blocks, JPEG_QUANT_MATRIX)
    M = h // block_size
    N = w // block_size

    # Извлечение коэффициентов выбранных полос (например, [11] или [10,11,12])
    freq_matrices = extract_frequency_matrices(blocks, selected_bands, M, N)
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
    wm_cells = compute_shifted_cells_symmetric(cells, watermark_bits, T, R)
    wm_freq_matrices = merge_cells(wm_cells, cell_height, cell_width, M, N)
    wm_quantized_blocks = replace_frequency_matrices_in_blocks(
        blocks, wm_freq_matrices, selected_bands, M, N
    )

    # Обратное квантование, IDCT и сборка итогового изображения
    # wm_dct_blocks = dequantize(wm_quantized_blocks, JPEG_QUANT_MATRIX)
    # wm_spatial_blocks = apply_idct(wm_dct_blocks)
    watermarked_img = merge_blocks(wm_quantized_blocks, cover_img.shape, block_size=block_size)

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

# if __name__ == "__main__":
#     # Пример: использование выбранных полос [10, 11, 12]
#     selected_bands = [11]
#
#     # 1) Встраивание водяного знака из watermark.jpg в Lena.jpg
#     wm_img, original_bits, T, cover_shape = embed_watermark(
#         cover_path="Lena.jpg",
#         watermark_path="BK.jpg",
#         block_size=8,
#         selected_bands=selected_bands,   # теперь передаётся список
#         cell_width=8,
#         cell_height=4
#     )
#     cv2.imwrite("Lena_watermarked.jpg", wm_img)
#     print("Watermarked image saved as Lena_watermarked.jpg")
#     print(f"Исходная битовая последовательность (длина): {len(original_bits)}")
#     # 2) Извлечение водяного знака
#     extracted_bits = extract_watermark(
#         watermarked_img=wm_img,
#         cover_shape=cover_shape,
#         block_size=8,
#         selected_bands=selected_bands,
#         cell_width=8,
#         cell_height=4,
#         key=key
#     )
#     print(f"Извлечённая битовая последовательность (длина): {len(extracted_bits)}")
#     print("Исходные биты:")
#     print(original_bits)
#     print("Извлечённые биты:")
#     print(extracted_bits)
#     num_errors = np.sum(original_bits != extracted_bits)
#     ber = num_errors / len(original_bits) * 100
#     print(f"Ошибок: {num_errors} из {len(original_bits)}  => BER = {ber:.2f}%")
#     if num_errors == 0:
#         print("Водяной знак извлечён без ошибок!")
#     else:
#         print("Есть расхождения между исходными и извлечёнными битами.")
#
#     # 3) Сохранение извлечённого водяного знака как бинарного изображения.
#     # Предполагаем, что длина водяного знака равна 128 бит, что соответствует размеру 8x16 пикселей.
#     extracted_watermark_img = (extracted_bits.reshape((8, 16)) * 255).astype(np.uint8)
#     cv2.imwrite("extracted_watermark.jpg", extracted_watermark_img)
#     print("Извлечённый водяной знак сохранён как extracted_watermark.jpg")
