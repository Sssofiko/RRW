from Constructing_robust_features import *


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
