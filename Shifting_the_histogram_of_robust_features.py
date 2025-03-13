from Constructing_robust_features import *
import matplotlib.pyplot as plt

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


def compute_shifted_cell(cell, w, t_r):
    """
    Вычисляет водяную версию исходной ячейки cell для данного параметра t_r
    и битового значения w (0 или 1) для этой ячейки.

    Параметры:
      cell: np.array размера (m, n) – исходная ячейка.
      w: целое число (0 или 1) – бит водяного знака для данной ячейки.
      t_r: целое число, параметр для данной частотной полосы.

    Возвращает:
      shifted_cell: np.array размера (m, n), представляющая водяную версию исходной ячейки cell.
    """
    m, n = cell.shape
    total = m * n
    # Приводим ячейку к вектору в порядке строки (row-major)
    flat = cell.flatten()
    # Создаем индексы от 0 до total-1
    eps = np.arange(total)
    # Вычисляем сдвиг для каждого элемента: floor((t_r + eps) / (m*n))
    shifts = np.floor((t_r + eps) / total).astype(int)
    # Фактор зависит от бита: (2*w - 1) будет равен +1, если w==1, и -1, если w==0.
    factor = 2 * w - 1
    # Для элементов, где (eps+1) нечётное, прибавляем; для чётных – вычитаем.
    odd_mask = ((eps + 1) % 2 == 1)
    flat_shifted = np.empty_like(flat)
    flat_shifted[odd_mask] = flat[odd_mask] + factor * shifts[odd_mask]
    flat_shifted[~odd_mask] = flat[~odd_mask] - factor * shifts[~odd_mask]
    # Приводим обратно к матричной форме
    return flat_shifted.reshape(m, n)
    # m, n = cell.shape
    # total = m * n
    # signs = np.array([1 if i % 2 == 0 else -1 for i in range(total)])
    # flat = cell.flatten()
    # lambda_orig = np.sum(flat * signs)
    # d = T if w == 1 else -T  # желаемое изменение
    # flat[0] += d
    # return flat.reshape(m, n)


def compute_shifted_cells(cells, w, T, R):
    """
    Применяет преобразование (вычисление x̃ₖᵣ(φₑ)) для всех ячеек во всех выбранных частотных полосах.

    Параметры:
      cells: np.array размера (R, L, m, n) – исходные ячейки, полученные из коэффициентов выбранных полос.
      w: np.array размера (L,) – вектор битов водяного знака для каждой ячейки (значения 0 или 1).
      T: целое число – порог, выбранный для встраивания (должно выполняться T > max(|λ(k)|)).
      R: целое число – число выбранных частотных полос.

    Возвращает:
      shifted_cells: np.array размера (R, L, m, n) – массив ячеек после применения целочисленного преобразования.
    """
    t = compute_t_r(T, R)  # вычисляем вектор параметров t для каждой полосы, размер R
    R_val, L, m, n = cells.shape
    shifted_cells = np.empty_like(cells)
    for r in range(R_val):
        for k in range(L):
            shifted_cells[r, k] = compute_shifted_cell(cells[r, k], w[k], t[r])
    return shifted_cells


# Функция, обратная divide_into_cells
def merge_cells(cells, cell_height, cell_width, M, N):
    """
    Восстанавливает исходную матрицу из ячеек, полученных функцией divide_into_cells.

    Параметры:
      cells: np.array размера (R, L, cell_height, cell_width),
             где R – число выбранных частотных полос,
                   L – общее число ячеек (L = (M//cell_height) * (N//cell_width)),
                   cell_height, cell_width – размеры каждой ячейки.
      M: итоговое число строк восстановленной матрицы (например, M = (число ячеек по вертикали)*cell_height).
      N: итоговое число столбцов восстановленной матрицы (например, N = (число ячеек по горизонтали)*cell_width).

    Возвращает:
      freq_matrix: np.array размера (R, M, N) – восстановленная матрица, которая соответствует
                   исходной матрице, разделённой функцией divide_into_cells.
    """
    R, L, m, n = cells.shape
    num_cells_vertical = M // cell_height
    num_cells_horizontal = N // cell_width

    freq_matrix = np.empty((R, M, N), dtype=cells.dtype)
    for r in range(R):
        cell_index = 0
        for i in range(0, num_cells_vertical * cell_height, cell_height):
            for j in range(0, num_cells_horizontal * cell_width, cell_width):
                freq_matrix[r, i:i + cell_height, j:j + cell_width] = cells[r, cell_index]
                cell_index += 1
    return freq_matrix


# Функция, обратная extract_frequency_matrices
def replace_frequency_matrices_in_blocks(blocks, freq_matrices, selected_bands, M, N):
    """
    Заменяет в оригинальных 8×8 блоках коэффициенты в выбранных частотных полосах на значения из freq_matrices.
    Остальные коэффициенты остаются без изменений.

    Параметры:
      blocks: np.array размера (M*N, 8, 8) – исходные блоки квантованных DCT коэффициентов.
      freq_matrices: np.array размера (R, M, N) – матрицы, где R = число выбранных полос.
                     Для каждого блока в позиции (i, j) матрицы содержит водяное значение для этой полосы.
      selected_bands: список номеров полос (от 1 до 64), выбранных согласно зигзаг-обходу.
      M: число блоков по вертикали (например, image_height // 8).
      N: число блоков по горизонтали (например, image_width // 8).

    Возвращает:
      modified_blocks: np.array размера (M*N, 8, 8) – блоки с заменёнными значениями для выбранных частотных полос.
    """
    # Создаем копию исходных блоков, чтобы не изменять оригинал
    modified_blocks = blocks.copy()
    # Получаем список координат по зигзаг-обходу для 8×8 блока
    zigzag = zigzag_indices(n=8)

    # Для каждой выбранной частотной полосы r
    for r, band in enumerate(selected_bands):
        # Определяем координаты (i, j) в блоке, соответствующие этой полосе (нумерация 1-64)
        i_coord, j_coord = zigzag[band - 1]
        # Проходим по всем блокам (расположенным в сетке размера M x N)
        for i in range(M):
            for j in range(N):
                block_idx = i * N + j
                # Заменяем коэффициент в позиции (i_coord, j_coord)
                modified_blocks[block_idx, i_coord, j_coord] = freq_matrices[r, i, j]

    return modified_blocks


# Пример применения:
if __name__ == "__main__":
    # Допустим, у нас есть изображение с размерами 512x512
    # Число блоков: M = 512 // 8 = 64, N = 512 // 8 = 64
    M = 512 // 8  # 64 блока по вертикали
    N = 512 // 8  # 64 блока по горизонтали
    num_blocks = M * N

    # Создадим псевдо-блоки (8×8) с целочисленными значениями, имитирующими квантованные DCT коэффициенты
    blocks = np.random.randint(-20, 20, (num_blocks, 8, 8))
    # Выбранные частотные полосы: R = 3, {13, 11, 20}
    selected_bands = [13, 11, 20]

    freq_matrices = extract_frequency_matrices(blocks, selected_bands, M, N)
    # три матрицы каждая из которых имеет размеры 64x64, как и расположение блоков в изображении.

    # Задаём размеры ячейки
    cell_height = 8  # высота ячейки (m)
    cell_width = 4  # ширина ячейки (n)

    # Число ячеек по вертикали: 64 // 8 = 8, по горизонтали: 64 // 4 = 16
    # Итого L = 8 * 16 = 128 ячеек на каждую матрицу.
    cells = divide_into_cells(freq_matrices, cell_height, cell_width)

    key = generate_random_bijection(cell_height, cell_width)

    # Вычисляем разностную статистику для каждой ячейки
    eta = compute_difference_statistics(cells, key)

    # Вычисляем робастный признак λ(k) для всех ячеек
    robust_features = compute_robust_features(eta)

    # Параметры примера
    R = len(selected_bands)  # число выбранных частотных полос
    L = M // cell_height * N // cell_width  # число ячеек в каждой матрице
    T = np.max(np.abs(robust_features)) + 1  # порог, соответствующий условию T > max(|λ(k)|)
    print(f'порог: {T}')

    # Случайно зададим биты водяного знака для каждой ячейки (0 или 1)
    w = np.random.randint(0, 2, size=(L,))

    print(f'водяной знак: {w[12]}')

    # print("Исходные ячейки (для каждой частотной полосы, L ячеек):")
    # for r in range(R):
    #     print(f"Полоса {r}:")
    #     print(cells[r])
    # print("\nБиты водяного знака для ячеек (w):")
    # print(w)

    # Вычисляем водяные ячейки
    wm_cells = compute_shifted_cells(cells, w, T, R)

    # print("\nЯчейки после применения целочисленного преобразования (x̃):")
    # for r in range(R):
    #     print(f"Полоса {r}:")
    #     print(shifted[r])

    wm_freq_matrices = merge_cells(wm_cells, cell_height, cell_width, M, N)

    wm_blocks = replace_frequency_matrices_in_blocks(blocks, wm_freq_matrices, selected_bands, M, N)

    # M = 512 // 8
    # N = 512 // 8
    # num_blocks = M * N
    #
    # # Исходные (псевдо)блоки квантованных DCT коэффициентов
    # blocks = np.random.randint(-20, 20, (num_blocks, 8, 8))
    # selected_bands = [13, 11, 20]
    #
    # # Извлекаем матрицы частот
    # freq_matrices = extract_frequency_matrices(blocks, selected_bands, M, N)
    #
    # cell_height = 8
    # cell_width = 4
    # cells = divide_into_cells(freq_matrices, cell_height, cell_width)
    #
    # # Генерируем случайный ключ (биекцию)
    # key = generate_random_bijection(cell_height, cell_width)
    #
    # # Считаем разностную статистику до встраивания
    # eta_before = compute_difference_statistics(cells, key)
    # robust_features_before = compute_robust_features(eta_before)
    #
    # # Определяем порог T
    # T = np.max(np.abs(robust_features_before)) + 1
    #
    # # Генерируем случайные биты водяного знака
    # L = (M // cell_height) * (N // cell_width)
    # w = np.random.randint(0, 2, size=(L,))
    #
    # R = len(selected_bands)
    #
    # # Выполняем целочисленное преобразование (встраивание)
    # wm_cells = compute_shifted_cells(cells, w, T, R)
    #
    # # Чтобы посмотреть, что получилось, пересчитаем робастные признаки после встраивания
    # eta_after = compute_difference_statistics(wm_cells, key)
    # robust_features_after = compute_robust_features(eta_after)
    #
    # # --- ПОСТРОЕНИЕ ГИСТОГРАММ ---
    #
    # plt.figure(figsize=(12, 5))
    #
    # # (a) Гистограмма робастных признаков до внедрения
    # plt.subplot(1, 2, 1)
    # plt.hist(robust_features_before, bins=30, color='orange')
    # plt.title("Histogram before watermarking")
    # plt.xlabel("Value")
    # plt.ylabel("Count")
    #
    # # (b) Гистограмма робастных признаков после внедрения
    # #    Покажем отрицательные и положительные значения разными цветами
    # plt.subplot(1, 2, 2)
    # neg_vals = robust_features_after[robust_features_after < 0]
    # pos_vals = robust_features_after[robust_features_after >= 0]
    # plt.hist(neg_vals, bins=30, color='red', alpha=0.5, label='bit-0-region')
    # plt.hist(pos_vals, bins=30, color='blue', alpha=0.5, label='bit-1-region')
    # plt.title("Histogram after watermarking")
    # plt.xlabel("Value")
    # plt.ylabel("Count")
    # plt.legend()
    #
    # plt.tight_layout()
    # plt.show()
