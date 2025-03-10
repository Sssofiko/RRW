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

