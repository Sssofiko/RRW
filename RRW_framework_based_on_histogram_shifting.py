import numpy as np
import cv2
import matplotlib.pyplot as plt


def robust_feature_extraction(quantized_blocks, selected_bands, key_s):
    """
    Вычисляет надежный признак lambda(k) для каждого квантованного блока
    как разностную статистику выбранных коэффициентов.

    Согласно статье:
      "The robust feature of P_k is denoted as lambda(k)"
    и формуле (8):
      eta(k) = sum_{i=1}^{mn} (-1)^(i-1) * x_k(phi_i)

    Параметры:
      quantized_blocks - массив квантованных блоков,
      selected_bands - список индексов выбранных коэффициентов (например, [(0,1)]),
      key_s - ключ для перестановки (например, np.array([0])).

    Возвращает:
      numpy.ndarray с вычисленными значениями lambda(k) для каждого блока.
    """
    features = []
    for block in quantized_blocks:
        coeffs = [block[u, v] for (u, v) in selected_bands]
        permuted = np.take(coeffs, key_s % len(coeffs))
        eta = sum(((-1) ** i) * permuted[i] for i in range(len(permuted)))
        features.append(eta)
    return np.array(features)


def plot_histogram(features, title):
    """
    Строит гистограмму значений features с указанным заголовком.

    Параметры:
      features - массив значений,
      title - заголовок графика.
    """
    plt.figure()
    plt.hist(features, bins=50, color='skyblue', edgecolor='black')
    plt.title(title)
    plt.xlabel('Robust feature value')
    plt.ylabel('Frequency')
    plt.show()


def choose_threshold(features, delta=5):
    """
    Выбирает порог T, удовлетворяющий условию T > max(|lambda(k)|).

    Параметры:
      features - массив значений lambda(k),
      delta - дополнительная величина (например, 5).

    Возвращает:
      Порог T.

    Цитата из статьи:
      "Then, a threshold T that satisfies T > max(|Λ|) is selected..."
    """
    return np.max(np.abs(features)) + delta


def embed_watermark_in_blocks(quantized_blocks, watermark_bits, selected_bands, T):
    """
    Встраивает водяной знак по формуле (6):
      lambda'(k)= lambda(k) + T, если w(k)=1,
      lambda'(k)= lambda(k) - T, если w(k)=0.

    Здесь для упрощения модифицируется только первый коэффициент из selected_bands.

    Параметры:
      quantized_blocks - массив квантованных блоков,
      watermark_bits - бинарная последовательность (число бит = число блоков),
      selected_bands - список выбранных индексов коэффициентов,
      T - пороговое значение.

    Возвращает:
      Массив блоков с обновлёнными DCT-коэффициентами.
    """
    updated_blocks = []
    for i, block in enumerate(quantized_blocks):
        mod_block = block.copy()
        u, v = selected_bands[0]
        if watermark_bits[i] == 1:
            mod_block[u, v] += T
        else:
            mod_block[u, v] -= T
        updated_blocks.append(mod_block)
    return np.array(updated_blocks)


def extract_watermark_from_blocks(quantized_blocks, selected_bands, key_s):
    """
    Извлекает водяной знак по формуле (7):
      w(k)= 1, если lambda'(k) >= 0; 0 иначе.

    Сначала для каждого блока вычисляется надежный признак lambda'(k),
    затем определяется бит.

    Параметры:
      quantized_blocks - массив квантованных блоков,
      selected_bands - список выбранных индексов коэффициентов,
      key_s - ключ для перестановки.

    Возвращает:
      Кортеж: (массив извлечённых битов, массив вычисленных lambda'(k)).
    """
    features = robust_feature_extraction(quantized_blocks, selected_bands, key_s)
    watermark = [1 if f >= 0 else 0 for f in features]
    return np.array(watermark), features


def simulate_RRW_framework(cover_quantized_blocks, cover_img_shape, watermark_path, selected_bands, key_s):
    """
    Симулирует процесс RRW framework based on histogram shifting для cover-изображения.

    Входные параметры:
      - cover_quantized_blocks: квантованные блоки cover-изображения (результат этапов 2.1),
      - cover_img_shape: размер исходного cover-изображения,
      - watermark_path: путь к файлу водяного знака ("watermark.jpg"),
      - selected_bands: список выбранных коэффициентов (например, [(0,1)]),
      - key_s: ключ для перестановки (например, np.array([0])).

    Этапы:
      1. Вычисление надежных признаков до встраивания и построение гистограммы.
      2. Выбор порога T, удовлетворяющего T > max(|lambda(k)|).
      3. Загрузка водяного знака и преобразование его в бинарную последовательность.
      4. Встраивание водяного знака (сдвиг коэффициентов по формуле (6)).
      5. Вычисление надежных признаков после встраивания и построение гистограммы.
      6. Извлечение водяного знака по формуле (7).

    Функция возвращает словарь с результатами (T, массивы признаков и водяной знак),
    который будет использоваться в main.py.
    """
    robust_before = robust_feature_extraction(cover_quantized_blocks, selected_bands, key_s)
    T = choose_threshold(robust_before, delta=5)
    # Загрузка водяного знака
    num_blocks = len(cover_quantized_blocks)
    side = int(np.sqrt(num_blocks))
    wm = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
    if wm is None:
        raise ValueError("Watermark image not found!")
    wm_resized = cv2.resize(wm, (side, side), interpolation=cv2.INTER_AREA)
    _, wm_binary = cv2.threshold(wm_resized, 128, 1, cv2.THRESH_BINARY)
    watermark_bits = wm_binary.flatten()
    if len(watermark_bits) != num_blocks:
        watermark_bits = np.resize(watermark_bits, num_blocks)
    updated_quantized_blocks = embed_watermark_in_blocks(cover_quantized_blocks, watermark_bits, selected_bands, T)
    robust_after = robust_feature_extraction(updated_quantized_blocks, selected_bands, key_s)
    extracted_wm, robust_extracted = extract_watermark_from_blocks(updated_quantized_blocks, selected_bands, key_s)

    return {
        "T": T,
        "robust_before": robust_before,
        "robust_after": robust_after,
        "watermark_bits": watermark_bits,
        "extracted_watermark": extracted_wm,
        "updated_quantized_blocks": updated_quantized_blocks,
        "cover_img_shape": cover_img_shape,
        "robust_extracted": robust_extracted
    }
