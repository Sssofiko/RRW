import numpy as np
from Review_of_JPEG_compression import zigzag_scan, inverse_zigzag_scan

def compute_t_r(T, R):
    """
    Вычисляет параметр t_r для каждой выбранной частотной полосы по формуле (13):
      t_r = (-1)^(r-1) * floor((T + r - 1) / R)
    Для удобства в Python (индексация с 0) используется:
      t[r] = (-1)**(r) * floor((T + r) / R),  r = 0,..., R-1.
    """
    t = np.zeros(R, dtype=int)
    for r in range(R):
        t[r] = int(np.floor((T + r) / R)) * ((-1) ** r)
    return t

def invert_permutation(perm):
    """
    Вычисляет обратную перестановку для заданного массива индексов.
    Принимает 1D-массив perm с числами от 1 до N и возвращает массив обратной перестановки.
    """
    inverse = np.empty_like(perm)
    perm0 = perm - 1  # перевод в 0-индексацию
    for i, p in enumerate(perm0):
        inverse[p] = i
    return inverse

def apply_integer_transformation(cells, watermark_bits, T, key):
    """
    Применяет целочисленную трансформацию (формула (12)) для встраивания бита водяного знака.

    Параметры:
      cells         - numpy.array размера (R, L, m, n): ячейки, полученные из выбранной частотной матрицы
      watermark_bits- массив длины L с битами водяного знака (0 или 1)
      T             - пороговое значение
      key           - приватный ключ (перестановка) размера (m*n,), содержащий числа от 1 до m*n

    Алгоритм:
      Для каждой полосы r = 0,..., R-1 и для каждой ячейки k = 0,..., L-1:
        1. Преобразует ячейку (m×n) в 1D-вектор.
        2. Переставляет элементы согласно key.
        3. Для каждого элемента с индексом epsilon (1-индексация, e = epsilon+1):
             если e нечётное:
               new_val = x + (2w(k)-1) * floor((t_r + e - 1)/(m*n))
             если e чётное:
               new_val = x - (2w(k)-1) * floor((t_r + e - 1)/(m*n))
        4. Инвертирует перестановку, чтобы вернуть элементы в исходный порядок.
    """
    R, L, m, n = cells.shape
    total = m * n
    t = compute_t_r(T, R)  # t_r для каждой полосы
    inv_key = invert_permutation(key)
    watermarked_cells = np.empty_like(cells)

    for r in range(R):
        for k in range(L):
            cell = cells[r, k].flatten()
            permuted = cell[key - 1].copy()
            for epsilon in range(total):
                e = epsilon + 1  # 1-индексация
                step = int(np.floor((t[r] + e - 1) / total))
                if e % 2 == 1:
                    permuted[epsilon] = permuted[epsilon] + (2 * watermark_bits[k] - 1) * step
                else:
                    permuted[epsilon] = permuted[epsilon] - (2 * watermark_bits[k] - 1) * step
            new_cell_flat = np.empty(total, dtype=permuted.dtype)
            new_cell_flat[inv_key] = permuted
            watermarked_cells[r, k] = new_cell_flat.reshape(m, n)
    return watermarked_cells

def compute_cell_difference_statistics(cells, key):
    """
    Вычисляет разностную статистику для каждой ячейки по формуле (10):
      η_r(k) = sum_{ε=1}^{m*n} (-1)^(ε-1) * x_k^r(φ_ε)
    """
    R, L, m, n = cells.shape
    total = m * n
    key0 = key - 1
    eta = np.zeros((R, L), dtype=int)
    for r in range(R):
        for k in range(L):
            cell = cells[r, k].flatten()
            permuted = cell[key0]
            stat = sum(((-1) ** epsilon) * permuted[epsilon] for epsilon in range(total))
            eta[r, k] = stat
    return eta

def compute_watermarked_robust_features(eta_hat):
    """
    Вычисляет итоговый водяной надежный признак для каждой группы ячеек по формуле (11):
      õλ(k) = sum_{r=1}^{R} (-1)^(r-1) * õη_r(k)
    """
    R, L = eta_hat.shape
    weights = np.array([(-1) ** r for r in range(R)])
    robust_features = np.sum(eta_hat * weights[:, None], axis=0)
    return robust_features

def shift_histogram_pipeline(cells, watermark_bits, T, key):
    """
    Встраивает водяной знак посредством сдвига гистограммы надежных признаков.

    Алгоритм:
      1. Применяет целочисленную трансформацию к каждой ячейке (формулы (12) и (13)).
      2. Вычисляет модифицированные разностные статистики (формула (10)).
      3. Объединяет статистики по формуле (11) для получения итогового watermarked robust feature.
    """
    watermarked_cells = apply_integer_transformation(cells, watermark_bits, T, key)
    eta_hat = compute_cell_difference_statistics(watermarked_cells, key)
    watermarked_robust_feat = compute_watermarked_robust_features(eta_hat)
    return watermarked_cells, watermarked_robust_feat

def update_quantized_dct_coefficients(quantized_blocks, freq_matrices, watermarked_cells, selected_bands, M, N):
    """
    Обновляет квантованные DCT коэффициенты для каждого блока по формуле (15):
      если f = σ_r, то d̂_{i,j}(f) = watermarked значение,
      иначе d̂_{i,j}(f) остаётся без изменений.
    """
    updated_blocks = quantized_blocks.copy()
    R = len(selected_bands)
    num_blocks = M * N
    updated_blocks_zigzag = []
    for idx in range(num_blocks):
        block = updated_blocks[idx]
        block_zigzag = zigzag_scan(block)
        i = idx // N
        j = idx % N
        for r in range(R):
            block_zigzag[selected_bands[r] - 1] = watermarked_cells[r, i * N + j]
        updated_block = inverse_zigzag_scan(block_zigzag)
        updated_blocks_zigzag.append(updated_block)
    updated_blocks_zigzag = np.array(updated_blocks_zigzag)
    return updated_blocks_zigzag
