import numpy as np


def zigzag_indices(n=8):
    """
    Возвращает список координат (i, j) для матрицы n×n, пронумерованных согласно зигзаг-обходу.
    """
    indices = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            # для чётных s идём сверху вниз
            for i in range(min(s, n - 1), max(-1, s - n), -1):
                j = s - i
                indices.append((i, j))
        else:
            # для нечётных s идём снизу вверх
            for i in range(max(0, s - n + 1), min(s, n - 1) + 1):
                j = s - i
                indices.append((i, j))
    return indices


def extract_frequency_matrices(blocks, selected_bands, M, N):
    """
    Извлекает коэффициенты квантованных DCT для выбранных частотных полос из каждого 8×8 блока.

    Параметры:
      blocks: np.array размерности (num_blocks, 8, 8) – набор блоков (например, после применения DCT и квантования).
      selected_bands: список номеров полос (1-64), выбранных пользователем согласно зигзаг-образному обходу.
      M: число блоков по вертикали (например, image_height // 8).
      N: число блоков по горизонтали (например, image_width // 8).

    Возвращает:
      np.array размерности (R, M, N), где R – число выбранных полос,
      а M×N – двумерное расположение блоков, соответствующее исходному изображению.
    """
    # Получаем список координат зигзаг-обхода для блока 8×8
    zigzag = zigzag_indices(n=8)

    # Для каждой выбранной полосы (номер 1-64) получаем соответствующую координату (i, j)
    selected_coords = [zigzag[band - 1] for band in selected_bands]

    # Извлекаем для каждой выбранной полосы коэффициенты из всех блоков
    coeff_vectors = []
    for (i, j) in selected_coords:
        vec = blocks[:, i, j]  # коэффициент с координатами (i, j) для каждого блока
        coeff_vectors.append(vec)

    # Преобразуем каждый вектор в матрицу размера (M, N)
    matrices = [vec.reshape(M, N) for vec in coeff_vectors]

    # Возвращаем np.array, где по оси 0 идут матрицы для каждой выбранной полосы
    return np.array(matrices)


def divide_into_cells(freq_matrices, cell_height, cell_width):
    """
    Делит каждую матрицу в freq_matrices на не перекрывающиеся ячейки размером cell_height x cell_width.

    Параметры:
      freq_matrices: np.array размера (R, M, N), где R – число выбранных частотных полос,
                     а M и N – размеры матрицы, полученной из расположения блоков.
      cell_height: высота ячейки (m)
      cell_width: ширина ячейки (n)

    Возвращает:
      cells: np.array размера (R, L, m, n), где L = floor(M/cell_height) * floor(N/cell_width)
             – число ячеек в каждой матрице.
    """
    R, M, N = freq_matrices.shape
    num_cells_vertical = M // cell_height
    num_cells_horizontal = N // cell_width
    L = num_cells_vertical * num_cells_horizontal

    # Инициализируем массив для ячеек
    cells = np.empty((R, L, cell_height, cell_width), dtype=freq_matrices.dtype)

    # Для каждой частотной полосы делим матрицу на ячейки
    for r in range(R):
        cell_index = 0
        for i in range(0, num_cells_vertical * cell_height, cell_height):
            for j in range(0, num_cells_horizontal * cell_width, cell_width):
                cells[r, cell_index] = freq_matrices[r, i:i + cell_height, j:j + cell_width]
                cell_index += 1

    return cells


def generate_random_bijection(m, n):
    """
    Генерирует случайную целочисленную биекцию (перестановку) элементов из множества {1, 2, ..., m*n}.

    Параметры:
      m: целое число, определяющее количество строк (например, высота ячейки)
      n: целое число, определяющее количество столбцов (например, ширина ячейки)

    Возвращает:
      bijection: np.array размера (m*n), содержащий случайную перестановку чисел от 1 до m*n.
    """
    total = m * n
    bijection = np.random.permutation(np.arange(1, total + 1))
    return bijection


def compute_difference_statistics(cells, key):
    """
    Вычисляет разностную статистику η_r(k) для каждой ячейки в cells.

    Параметры:
      cells: np.array размера (R, L, m, n), где R – число выбранных частотных полос,
             L – число ячеек в каждой полосе, m и n – размеры каждой ячейки.
      key: np.array размера (m*n,), содержащий случайную перестановку чисел от 1 до m*n.
           Эта перестановка задаёт порядок обхода элементов ячейки (закрытый ключ).

    Возвращает:
      eta: np.array размера (R, L), где каждый элемент соответствует разностной статистике
           для соответствующей ячейки (для большей ясности, элементы в eta[k], где k = (1,...,L),
           представляют собой одномерный массив длины L).
    """
    R, L, m, n = cells.shape
    total = m * n

    # Формируем вектор знаков: для epsilon=1 даем знак +1, для epsilon=2 знак -1 и так далее.
    signs = np.array([1 if i % 2 == 0 else -1 for i in range(total)])

    # Преобразуем каждую ячейку в вектор длины m*n (сохраняя порядок по строкам)
    cells_flat = cells.reshape(R, L, total)

    # Ключ key содержит числа от 1 до m*n, для индексации в Python вычтем 1.
    key_idx = key - 1  # теперь key_idx содержит индексы от 0 до m*n-1

    # Переставляем элементы каждого вектора в порядке, заданном ключом.
    # Для каждого элемента cells_flat[r, k, :] применяем перестановку key_idx.
    permuted_cells = cells_flat[:, :, key_idx]

    # Вычисляем разностную статистику для каждой ячейки как скалярное произведение
    # вектора переставленных элементов и вектора знаков.
    eta = np.sum(permuted_cells * signs, axis=2)

    return eta


def compute_robust_features(eta):
    """
    Вычисляет робастный признак λ(k) для каждой группы ячеек (k) на основе разностных статистик.

    Параметры:
      eta: np.array размера (R, L), где R – число выбранных частотных полос,
           а L – число ячеек (например, из cells). Элемент eta[r, k] соответствует разностной
           статистике η_r(k) для r-й частотной полосы в k-й ячейке.

    Возвращает:
      lambda_k: np.array размера (L), содержащий робастный признак λ(k) для каждой группы ячеек.
                При этом знак каждого слагаемого определяется по формуле (-1)^(r-1).
    """
    R, L = eta.shape
    # Создаем вектор весов: для r = 1, 2, 3, ... веса будут 1, -1, 1, -1, ...
    # (в Python нумерация с 0, поэтому используем (-1)**(np.arange(R)))
    weights = (-1) ** (np.arange(R))

    # Вычисляем λ(k) для каждой ячейки k: суммируем произведение соответствующих весов и η_r(k) по r.
    lambda_k = (eta * weights[:, None]).sum(axis=0)

    return lambda_k

# Пример использования:
if __name__ == "__main__":
    # Допустим, у нас есть изображение с размерами 256x320
    # Число блоков: M = 256 // 8 = 32, N = 320 // 8 = 40
    M = 256 // 8  # 32 блоков по вертикали
    N = 320 // 8  # 40 блоков по горизонтали
    num_blocks = M * N

    # Создадим псевдо-блоки (8×8) с целочисленными значениями, имитирующими квантованные DCT коэффициенты
    blocks = np.random.randint(-20, 20, (num_blocks, 8, 8))
    # Выбранные частотные полосы: R = 3, {13, 11, 20}
    selected_bands = [13, 11, 20]

    freq_matrices = extract_frequency_matrices(blocks, selected_bands, M, N)
    print("Форма выходного массива:", freq_matrices.shape)
    # Вывод: (3, 32, 40) – три матрицы, соответствующие выбранным частотным полосам,
    # каждая из которых имеет размеры 32x40, как и расположение блоков в изображении.

    # Пример вывода коэффициентов для первой выбранной полосы (номер 13)
    print("Коэффициенты для полосы 13 (первая матрица):")
    print(freq_matrices[0])

    # Задаём размеры ячейки:
    cell_height = 4  # высота ячейки (m)
    cell_width = 6  # ширина ячейки (n) – согласно условию cell_width = 6

    # Число ячеек по вертикали: 32 // 4 = 8, по горизонтали: 40 // 6 = 6
    # Итого L = 8 * 6 = 48 ячеек на каждую матрицу.
    cells = divide_into_cells(freq_matrices, cell_height, cell_width)

    print("\nФорма выходного массива ячеек:", cells.shape)
    # Ожидается (3, 48, 4, 6)

    # Пример: вывод 10-й ячейки из первой матрицы (первой выбранной частотной полосы)
    print("Ячейка №1 для первой частотной полосы:")
    print(cells[0, 0])

    key = generate_random_bijection(cell_height, cell_width)
    print("\nСлучайная целочисленная биекция (закрытый ключ):")
    print(key)

    # Вычисляем разностную статистику для каждой ячейки
    eta = compute_difference_statistics(cells, key)
    print("\nРазностная статистика η (размер (R, L)):")
    print(eta)

    # Вычисляем робастный признак λ(k) для всех ячеек:
    robust_features = compute_robust_features(eta)
    print("\nРазностные статистики (eta) для первой полосы, второй, третьей:")
    print(eta)
    print("\nРобастный признак λ для каждой ячейки:")
    print(robust_features)
