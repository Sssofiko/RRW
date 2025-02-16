import cv2
import numpy as np
import json

def convert_to_YCbCr(image_path):
    """
    Loads an image and converts it to YCbCr color space.

    Parameters:
        image_path (str): Path to the cover image.

    Returns:
        ycbcr_img (numpy.ndarray): YCbCr image.
        Y_channel (numpy.ndarray): Luminance (Y) channel.
    """
    img = cv2.imread(image_path)  # Load image in BGR format
    ycbcr_img = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)  # Convert to YCbCr
    Y_channel = ycbcr_img[:, :, 0]  # Extract Y-channel
    return ycbcr_img, Y_channel

def preprocess_watermark(watermark_path, size):
    """
    Loads a watermark image, converts it to grayscale, and binarizes it.

    Parameters:
        watermark_path (str): Path to the watermark image.
        size (tuple): Desired size for resizing the watermark (width, height).

    Returns:
        binary_watermark (numpy.ndarray): Binary representation of the watermark.
    """
    img = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
    img_resized = cv2.resize(img, size, interpolation=cv2.INTER_AREA)  # Resize to fit
    _, binary_watermark = cv2.threshold(img_resized, 128, 1, cv2.THRESH_BINARY)  # Binarize
    return binary_watermark

def divide_into_8x8_blocks(Y_channel):
    """
    Splits the Y-channel into 8x8 non-overlapping blocks.

    Parameters:
        Y_channel (numpy.ndarray): Luminance channel of the image.

    Returns:
        blocks (list of numpy.ndarray): List of 8x8 blocks.
    """
    h, w = Y_channel.shape
    blocks = [Y_channel[i:i+8, j:j+8] for i in range(0, h, 8) for j in range(0, w, 8)]
    return blocks

def apply_DCT(blocks):
    """
    Applies 2D Discrete Cosine Transform (DCT) to each 8x8 block.

    Parameters:
        blocks (list of numpy.ndarray): List of 8x8 image blocks.

    Returns:
        dct_blocks (list of numpy.ndarray): DCT-transformed blocks.
    """
    return [cv2.dct(block.astype(np.float32)) for block in blocks]


# Standard JPEG quantization table for the luminance (Y) channel
JPEG_QUANTIZATION_TABLE = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)

# Update the quantization function to use the default table
def quantize_DCT(dct_blocks):
    """
    Quantizes the DCT coefficients using the standard JPEG quantization table.

    Parameters:
        dct_blocks (list of numpy.ndarray): List of DCT-transformed blocks.

    Returns:
        quantized_blocks (list of numpy.ndarray): Quantized DCT coefficient blocks.
    """
    return [np.round(block / JPEG_QUANTIZATION_TABLE).astype(np.int32) for block in dct_blocks]

def select_frequency_bands(num_bands):
    """
    Selects num_bands frequency bands from the low-to-mid frequency range.

    Parameters:
        num_bands (int): Number of frequency bands to select.

    Returns:
        selected_bands (list): Indices of selected frequency bands.
    """
    low_freq_bands = [(u, v) for u in range(8) for v in range(8) if (u, v) != (0, 0)]  # Exclude DC coefficient
    return low_freq_bands[:num_bands]  # Select the first num_bands frequencies

def compute_difference_statistic(block, selected_bands, key_s):
    """
    Computes the difference statistic eta_r(k) for a given block using selected frequency bands.

    Parameters:
        block (numpy.ndarray): Quantized DCT block.
        selected_bands (list): List of selected frequency bands.
        key_s (numpy.ndarray): Random mapping key.

    Returns:
        eta_values (list): Difference statistics for the selected bands.
    """
    coefficients = np.array([block[u, v] for (u, v) in selected_bands])  # Convert to NumPy array
    shuffled_coeffs = np.take(coefficients, key_s % len(coefficients))  # Use modulo to prevent out-of-bounds indexing
    eta_values = [sum((-1)**i * shuffled_coeffs[i] for i in range(len(shuffled_coeffs)))]
    return eta_values

def compute_robust_feature(eta_values):
    """
    Computes the final robust feature lambda(k) by summing eta_r(k) values with alternating signs.

    Parameters:
        eta_values (list): List of difference statistics.

    Returns:
        lambda_k (int): Robust feature value.
    """
    return sum((-1)**(r-1) * eta_values[r] for r in range(len(eta_values)))

def shift_histogram_for_watermark(robust_features, watermark_bits, threshold_t):
    """
    Shifts the histogram of robust features to embed watermark bits.

    Parameters:
        robust_features (list of int): List of computed robust features.
        watermark_bits (list of int): Binary watermark sequence (0s and 1s).
        threshold_t (int): Threshold for shifting.

    Returns:
        modified_features (list of int): Updated robust features after embedding.
    """
    modified_features = []
    for k in range(len(robust_features)):
        if watermark_bits[k] == 1:
            modified_features.append(robust_features[k] + threshold_t)
        else:
            modified_features.append(robust_features[k] - threshold_t)
    return modified_features

def update_dct_coefficients(quantized_dct, modified_features, selected_bands):
    """
    Обновляет коэффициенты DCT с учетом встроенного водяного знака.

    Исправление: Учитываем, что изменения коэффициентов DCT не должны выходить за допустимый диапазон.

    Parameters:
        quantized_dct (list of numpy.ndarray): Список квантованных DCT-блоков.
        modified_features (list of int): Измененные признаки для встраивания.
        selected_bands (list): Индексы выбранных частотных полос.

    Returns:
        updated_dct (list of numpy.ndarray): Обновленные DCT-блоки.
    """
    updated_dct = quantized_dct.copy()
    for i, block in enumerate(updated_dct):
        if i < len(modified_features):
            feature_value = modified_features[i]
            for band_index, (u, v) in enumerate(selected_bands):
                if band_index == 0:
                    block[u, v] = np.clip(feature_value, -127, 127)  # Ограничиваем диапазон значений
    return updated_dct


def dequantize_dct(quantized_dct):
    """
    Dequantizes the modified DCT coefficients using the standard JPEG quantization table.

    Parameters:
        quantized_dct (list of numpy.ndarray): List of quantized DCT blocks.

    Returns:
        dequantized_dct (list of numpy.ndarray): Dequantized DCT blocks.
    """
    return [block * JPEG_QUANTIZATION_TABLE for block in quantized_dct]

def apply_idct(dequantized_dct):
    """
    Applies the inverse 2D Discrete Cosine Transform (IDCT) to reconstruct image blocks.

    Parameters:
        dequantized_dct (list of numpy.ndarray): List of dequantized DCT blocks.

    Returns:
        idct_blocks (list of numpy.ndarray): Reconstructed 8x8 spatial domain blocks.
    """
    return [cv2.idct(block.astype(np.float32)) for block in dequantized_dct]

def merge_blocks(idct_blocks, image_shape):
    """
    Reconstructs the Y-channel from processed 8x8 blocks.

    Parameters:
        idct_blocks (list of numpy.ndarray): List of 8x8 IDCT-processed blocks.
        image_shape (tuple): Original Y-channel shape (height, width).

    Returns:
        reconstructed_y (numpy.ndarray): Reconstructed Y-channel.
    """
    h, w = image_shape
    reconstructed_y = np.zeros((h, w), dtype=np.uint8)

    index = 0
    for i in range(0, h, 8):
        for j in range(0, w, 8):
            reconstructed_y[i:i+8, j:j+8] = np.clip(idct_blocks[index], 0, 255)
            index += 1

    return reconstructed_y

def merge_ycbcr(y_channel, cb_channel, cr_channel):
    """
    Merges the modified Y-channel with the original Cb and Cr channels.

    Parameters:
        y_channel (numpy.ndarray): Modified Y-channel.
        cb_channel (numpy.ndarray): Original Cb channel.
        cr_channel (numpy.ndarray): Original Cr channel.

    Returns:
        merged_image (numpy.ndarray): Reconstructed YCbCr image.
    """
    ycbcr_img = np.stack((y_channel, cb_channel, cr_channel), axis=-1)
    return cv2.cvtColor(ycbcr_img, cv2.COLOR_YCrCb2BGR)

def save_as_jpeg(image, output_path, metadata):
    """
    Saves the final image as a JPEG file with embedded metadata.

    Parameters:
        image (numpy.ndarray): Watermarked image.
        output_path (str): Path to save the watermarked JPEG image.
        metadata (dict): Metadata containing threshold, frequency bands, and key.
    """
    cv2.imwrite(output_path, image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    # Metadata storage can be implemented separately, e.g., using JSON or EXIF data.

def save_metadata(metadata, metadata_path):
    """
    Saves metadata (threshold, frequency bands, key) as a JSON file.

    Parameters:
        metadata (dict): Metadata containing threshold, frequency bands, and key.
        metadata_path (str): Path to save the metadata file.
    """
    with open(metadata_path, "w") as file:
        json.dump(metadata, file)

def embed_watermark(cover_image_path, watermark_image_path, output_image_path, metadata_path, threshold_t, num_bands, key_s):
    """
    Embeds a watermark into a JPEG image.

    Parameters:
        cover_image_path (str): Path to the cover JPEG image.
        watermark_image_path (str): Path to the watermark JPEG image.
        output_image_path (str): Path to save the watermarked JPEG image.
        metadata_path (str): Path to save metadata for extraction.
        threshold_t (int): Threshold for histogram shifting.
        num_bands (int): Number of frequency bands to use for watermarking.
        key_s (numpy.ndarray): Random mapping key.
    """
    # Step 1: Load and preprocess images
    cover_ycbcr, Y_channel = convert_to_YCbCr(cover_image_path)
    watermark_binary = preprocess_watermark(watermark_image_path, (Y_channel.shape[1] // 8, Y_channel.shape[0] // 8))

    # Step 2: JPEG compression simulation
    blocks = divide_into_8x8_blocks(Y_channel)
    dct_blocks = apply_DCT(blocks)
    quantized_dct = quantize_DCT(dct_blocks)

    # Step 3: Construct robust features
    selected_bands = select_frequency_bands(num_bands)
    robust_features = []
    for i, block in enumerate(quantized_dct):
        eta_values = compute_difference_statistic(block, selected_bands, key_s)
        robust_features.append(compute_robust_feature(eta_values))

    # Step 4: Embed the watermark using histogram shifting
    modified_features = shift_histogram_for_watermark(robust_features, watermark_binary.flatten(), threshold_t)
    updated_dct = update_dct_coefficients(quantized_dct, modified_features, selected_bands)

    # Step 5: Reconstruct the JPEG image
    dequantized_dct = dequantize_dct(updated_dct)
    idct_blocks = apply_idct(dequantized_dct)
    reconstructed_Y = merge_blocks(idct_blocks, Y_channel.shape)
    watermarked_image = merge_ycbcr(reconstructed_Y, cover_ycbcr[:, :, 1], cover_ycbcr[:, :, 2])

    # Save the final JPEG image
    save_as_jpeg(watermarked_image, output_image_path, metadata={"T": threshold_t, "bands": selected_bands, "key": key_s.tolist()})

    # Save metadata
    save_metadata({"T": threshold_t, "bands": selected_bands, "key": key_s.tolist()}, metadata_path)

    print(f"Watermark embedded successfully. Saved at {output_image_path}")

def load_metadata(metadata_path):
    """
    Loads metadata from a JSON file.

    Parameters:
        metadata_path (str): Path to the metadata file.

    Returns:
        metadata (dict): Dictionary containing threshold, frequency bands, and key.
    """
    with open(metadata_path, "r") as file:
        return json.load(file)

def extract_robust_features(watermarked_image_path, selected_bands, key_s):
    """
    Extracts robust features from the watermarked JPEG image.

    Parameters:
        watermarked_image_path (str): Path to the watermarked JPEG image.
        selected_bands (list): Indices of selected frequency bands.
        key_s (numpy.ndarray): Random mapping key.

    Returns:
        extracted_features (list): Extracted robust features from the image.
    """
    _, Y_channel = convert_to_YCbCr(watermarked_image_path)
    blocks = divide_into_8x8_blocks(Y_channel)
    dct_blocks = apply_DCT(blocks)
    quantized_dct = quantize_DCT(dct_blocks)

    extracted_features = []
    for i, block in enumerate(quantized_dct):
        eta_values = compute_difference_statistic(block, selected_bands, key_s)
        extracted_features.append(compute_robust_feature(eta_values))

    return extracted_features

def extract_watermark(robust_features, threshold_t):
    """
    Извлекает бинарный водяной знак из извлеченных признаков.

    Исправление: Используем порог относительно среднего значения.

    Parameters:
        robust_features (list of int): Извлеченные признаки.
        threshold_t (int): Пороговое значение.

    Returns:
        extracted_watermark (list of int): Извлеченный водяной знак (0 и 1).
    """
    avg_feature = np.mean(robust_features)  # Усредняем признаки
    return [1 if feature >= avg_feature else 0 for feature in robust_features]

def restore_original_features(robust_features, threshold_t):
    """
    Restores the original robust features by reversing histogram shifting.

    Parameters:
        robust_features (list of int): Extracted robust features.
        threshold_t (int): Threshold used in histogram shifting.

    Returns:
        original_features (list of int): Restored robust features.
    """
    return [feature - threshold_t if feature >= 0 else feature + threshold_t for feature in robust_features]

def restore_original_dct(quantized_dct, original_features, selected_bands):
    """
    Восстанавливает оригинальные коэффициенты DCT после удаления водяного знака.

    Исправление: Добавляем обработку случаев, когда исходные значения выходят за диапазон.

    Parameters:
        quantized_dct (list of numpy.ndarray): Квантованные DCT-блоки.
        original_features (list of int): Восстановленные оригинальные признаки.
        selected_bands (list): Индексы выбранных частотных полос.

    Returns:
        restored_dct (list of numpy.ndarray): Восстановленные DCT-блоки.
    """
    restored_dct = quantized_dct.copy()
    for i, block in enumerate(restored_dct):
        if i < len(original_features):
            feature_value = original_features[i]
            for band_index, (u, v) in enumerate(selected_bands):
                if band_index == 0:
                    block[u, v] = np.clip(feature_value, -127, 127)  # Ограничиваем значения
    return restored_dct

def reconstruct_original_image(watermarked_image_path, metadata_path, output_image_path, extracted_watermark_path):
    """
    Restores the original JPEG image and extracts the embedded watermark.

    Parameters:
        watermarked_image_path (str): Path to the watermarked JPEG image.
        metadata_path (str): Path to the metadata file.
        output_image_path (str): Path to save the restored JPEG image.
        extracted_watermark_path (str): Path to save the extracted watermark.
    """
    metadata = load_metadata(metadata_path)
    threshold_t = metadata["T"]
    selected_bands = metadata["bands"]
    key_s = np.array(metadata["key"])

    extracted_features = extract_robust_features(watermarked_image_path, selected_bands, key_s)
    extracted_watermark = extract_watermark(extracted_features, threshold_t)
    print(extracted_watermark)
    # Save the extracted watermark
    watermark_size = (64, 64)
    # Assuming original size
    save_extracted_watermark(extracted_watermark, watermark_size, extracted_watermark_path)

    original_features = restore_original_features(extracted_features, threshold_t)

    # Load watermarked image to get its DCT coefficients
    cover_ycbcr, Y_channel = convert_to_YCbCr(watermarked_image_path)
    blocks = divide_into_8x8_blocks(Y_channel)
    dct_blocks = apply_DCT(blocks)
    quantized_dct = quantize_DCT(dct_blocks)

    # Restore original DCT coefficients
    restored_dct = restore_original_dct(quantized_dct, original_features, selected_bands)

    # Reconstruct the original image
    dequantized_dct = dequantize_dct(restored_dct)
    idct_blocks = apply_idct(dequantized_dct)
    restored_Y = merge_blocks(idct_blocks, Y_channel.shape)
    original_image = merge_ycbcr(restored_Y, cover_ycbcr[:, :, 1], cover_ycbcr[:, :, 2])

    save_as_jpeg(original_image, output_image_path, metadata=None)
    print(f"Original image restored successfully. Saved at {output_image_path}")


def save_extracted_watermark(watermark_bits, size, output_path):
    """
    Сохраняет извлеченный водяной знак в виде изображения.

    Исправление: Убедимся, что размер корректный перед reshaping.

    Parameters:
        watermark_bits (list of int): Бинарный водяной знак.
        size (tuple): Размер (width, height).
        output_path (str): Путь сохранения.
    """
    try:
        watermark_array = np.array(watermark_bits, dtype=np.uint8)
        if watermark_array.size != size[0] * size[1]:
            raise ValueError(f"Размер watermark_bits ({watermark_array.size}) не совпадает с {size}")

        watermark_array = watermark_array.reshape(size)
        watermark_image = (watermark_array * 255).astype(np.uint8)
        cv2.imwrite(output_path, watermark_image)
        print(f"Извлеченный водяной знак сохранен: {output_path}")
    except Exception as e:
        print(f"Ошибка при сохранении водяного знака: {e}")


if __name__ == "__main__":
    embed_watermark(
        cover_image_path="Lena.jpg",
        watermark_image_path="watermark.jpg",
        output_image_path="watermarked.jpg",
        metadata_path="metadata.json",
        threshold_t=21,
        num_bands=10,
        key_s=np.random.permutation(10)
    )

    reconstruct_original_image(
        watermarked_image_path="watermarked.jpg",
        metadata_path="metadata.json",
        output_image_path="restored.jpg",
        extracted_watermark_path="extracted_watermark.jpg"
    )

