import numpy as np

def mse(image1, image2):
    """
    Вычисляет среднеквадратичную ошибку (MSE) между двумя изображениями.

    Параметры:
      image1, image2: np.array - два изображения для сравнения.

    Возвращает:
      mse_value: Среднеквадратичная ошибка.
    """
    if image1.shape != image2.shape:
        raise ValueError("Изображения должны быть одинакового размера")

    mse_value = np.mean((image1 - image2) ** 2)
    return mse_value

def psnr(image1, image2):
    """
    Вычисляет пиковое отношение сигнал/шум (PSNR) между двумя изображениями.

    Параметры:
      image1, image2: np.array - два изображения для сравнения.

    Возвращает:
      psnr_value: Пиковое отношение сигнал/шум.
    """
    mse_value = mse(image1, image2)
    if mse_value == 0:
        return float('inf')

    max_pixel = 255.0
    psnr_value = 20 * np.log10(max_pixel / np.sqrt(mse_value))
    return psnr_value

def rmse(image1, image2):
    """
    Вычисляет корень из средней квадратичной ошибки (RMSE) между двумя изображениями.

    Параметры:
      image1, image2: np.array - два изображения для сравнения.

    Возвращает:
      rmse_value: Корень из средней квадратичной ошибки.
    """
    mse_value = mse(image1, image2)
    rmse_value = np.sqrt(mse_value)
    return rmse_value

def ssim(image1, image2):
    """
    Вычисляет структурное сходство (SSIM) между двумя изображениями.

    Параметры:
      image1, image2: np.array - два изображения для сравнения.

    Возвращает:
      ssim_value: Структурное сходство.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    mu1 = np.mean(image1)
    mu2 = np.mean(image2)
    sigma1 = np.var(image1)
    sigma2 = np.var(image2)
    sigma12 = np.cov(image1.flatten(), image2.flatten())[0, 1]

    numerator = (2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1 ** 2 + mu2 ** 2 + C1) * (sigma1 + sigma2 + C2)

    ssim_value = numerator / denominator
    return ssim_value

def robustness(image1, image2):
    """
    Вычисляет робастность с помощью BER (битовая ошибка) и NCC (нормированная кросс-корреляция).

    Параметры:
      image1, image2: np.array - два изображения для сравнения.

    Возвращает:
      ber: Битовая ошибка (Bit Error Rate).
      ncc: Нормированная кросс-корреляция (Normalized Cross-Correlation).
    """
    # Преобразуем изображения в двоичный вид
    image1_binary = (image1 > 128).astype(int)
    image2_binary = (image2 > 128).astype(int)

    # Вычисляем BER (битовая ошибка)
    ber = np.sum(image1_binary != image2_binary) / float(image1.size)

    # Вычисляем NCC (нормированная кросс-корреляция)
    image1_flat = image1.flatten()
    image2_flat = image2.flatten()

    numerator = np.dot(image1_flat - np.mean(image1_flat), image2_flat - np.mean(image2_flat))
    denominator = np.sqrt(np.sum((image1_flat - np.mean(image1_flat)) ** 2) * np.sum((image2_flat - np.mean(image2_flat)) ** 2))

    ncc = numerator / denominator if denominator != 0 else 0

    return ber, ncc



# import numpy as np, math
# # Значение среднеквадратичного отклонения
# def calculate_MSE(img, container_img_before, container_img_after):
#     # Проверка на совпадение размеров изображений
#     if container_img_before.shape != container_img_after.shape:
#         raise ValueError("Размеры изображений должны быть одинаковыми.")
#
#     # Вычисление среднеквадратичного отклонения
#     squared_error_sum = np.sum(np.square(container_img_before - container_img_after))
#     height, width = img.size
#     num_pixels = height * width
#
#     MSE = squared_error_sum / num_pixels
#
#     return MSE
#
#
# # Пиковое отношение сигнал-шум
# def calculate_PSNR(MSE):
#     if MSE == 0:
#         return float('inf')  # Изображения идентичны, PSNR бесконечно большое
#     max_pixel_value = 255  # Максимальное значение пикселя для 8-битного изображения
#     PSNR = 10 * math.log10((max_pixel_value ** 2) / MSE)
#     return PSNR
#
#
# # Квадратный корень из среднеквадратичного отклонения
# def calculate_RMSE(MSE):
#     RMSE = math.sqrt(MSE)
#     return RMSE
#
#
# # Индекс структурного сходства
# def calculate_SSIM(container_img, embedded_img, K1=0.01, K2=0.03):
#     # Константы
#     L = 255  # динамический диапазон значений пикселей
#     c1 = (K1 * L) ** 2
#     c2 = (K2 * L) ** 2
#
#     # Вычисление среднего значения пикселей
#     mu_P = np.mean(container_img)
#     mu_S = np.mean(embedded_img)
#
#     # Вычисление дисперсий
#     sigma_P_squared = np.var(container_img)
#     sigma_S_squared = np.var(embedded_img)
#
#     # Вычисление ковариации
#     sigma_PS = np.cov(container_img.flatten(), embedded_img.flatten())[0, 1]
#
#     # SSIM
#     numerator = (2 * mu_P * mu_S + c1) * (2 * sigma_PS + c2)
#     denominator = (mu_P ** 2 + mu_S ** 2 + c1) * (sigma_P_squared + sigma_S_squared + c2)
#     SSIM = numerator / denominator
#
#     return SSIM
#
# def robustness(message, mod_extracted_text):
#     # Преобразование сообщения в бинарный формат
#     binary_message = ''.join(format(ord(char), '08b') for char in message)
#
#     # Преобразование извлеченного текста из модифицированного изображения в бинарный формат
#     binary_mod_extracted_text = ''.join(format(ord(char), '08b') for char in mod_extracted_text)
#
#     # Вычисление интенсивности битовых ошибок
#     num_errors = sum(
#         1 for bit_msg, bit_mod_ext in zip(binary_message, binary_mod_extracted_text) if bit_msg != bit_mod_ext)
#     bit_error_rate = num_errors / len(binary_message)
#
#     # Вычисляем сумму произведений соответствующих битов в сообщениях
#     numerator = sum(
#         int(bit_msg) * int(bit_mod_ext) for bit_msg, bit_mod_ext in zip(binary_message, binary_mod_extracted_text))
#
#     # Вычисляем суммы квадратов значений битов в сообщениях
#     sum_message_squared = sum(int(bit) ** 2 for bit in binary_message)
#     sum_mod_extracted_squared = sum(int(bit) ** 2 for bit in binary_mod_extracted_text)
#
#     # Вычисляем корни из сумм квадратов
#     sqrt_sum_message_squared = sum_message_squared ** 0.5
#     sqrt_sum_mod_extracted_squared = sum_mod_extracted_squared ** 0.5
#
#     # Нормализуем сумму произведений на корень из произведения сумм квадратов
#     # Вычисляем нормализованную кросс-корреляцию
#     NCC = numerator / (sqrt_sum_message_squared * sqrt_sum_mod_extracted_squared)
#
#     return bit_error_rate, NCC
