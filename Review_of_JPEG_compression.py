import numpy as np
import cv2
from scipy.fftpack import dct, idct
import heapq
import matplotlib.pyplot as plt

# --- Стандартная матрица квантования для яркостного канала JPEG ---
# --- Матрица квантования с более высокими значениями ---
JPEG_QUANT_MATRIX = np.array([
    [16, 11, 10, 16, 24, 40, 51, 61],
    [12, 12, 14, 19, 26, 58, 60, 55],
    [14, 13, 16, 24, 40, 57, 69, 56],
    [14, 17, 22, 29, 51, 87, 80, 62],
    [18, 22, 37, 56, 68, 109, 103, 77],
    [24, 35, 55, 64, 81, 104, 113, 92],
    [49, 64, 78, 87, 103, 121, 120, 101],
    [72, 92, 95, 98, 112, 100, 103, 99]
], dtype=np.float32)


# --- Функции для разбиения и обратной сборки блоков ---
def split_into_blocks(img, block_size=8):
    h, w = img.shape
    blocks = [img[i:i+block_size, j:j+block_size]
              for i in range(0, h, block_size) for j in range(0, w, block_size)]
    return np.array(blocks)

def merge_blocks(blocks, img_shape, block_size=8):
    h, w = img_shape
    img_reconstructed = np.zeros((h, w))
    idx = 0
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            img_reconstructed[i:i+block_size, j:j+block_size] = blocks[idx]
            idx += 1
    return np.clip(img_reconstructed, 0, 255).astype(np.uint8)

# --- Функции для DCT, квантования, деквантования и IDCT ---
def apply_dct(blocks):
    return np.array([dct(dct(block.T, norm='ortho').T, norm='ortho') for block in blocks])


def quantize(blocks, quant_matrix):
    return np.round(blocks / quant_matrix).astype(int)

def dequantize(blocks, quant_matrix):
    return (blocks * quant_matrix).astype(float)

def apply_idct(blocks):
    return np.array([idct(idct(block.T, norm='ortho').T, norm='ortho') for block in blocks])

# --- Функции для зигзагообразной перестановки ---
def zigzag_scan(block):
    h, w = block.shape
    result = np.empty(h * w, dtype=block.dtype)
    index = -1
    bound = h + w - 1
    for s in range(bound):
        if s % 2 == 0:
            x = min(s, h - 1)
            y = s - x
            while x >= 0 and y < w:
                index += 1
                result[index] = block[x, y]
                x -= 1
                y += 1
        else:
            y = min(s, w - 1)
            x = s - y
            while y >= 0 and x < h:
                index += 1
                result[index] = block[x, y]
                x += 1
                y -= 1
    return result

def inverse_zigzag_scan(array, block_size=8):
    block = np.empty((block_size, block_size), dtype=array.dtype)
    index = -1
    bound = block_size + block_size - 1
    for s in range(bound):
        if s % 2 == 0:
            x = min(s, block_size - 1)
            y = s - x
            while x >= 0 and y < block_size:
                index += 1
                block[x, y] = array[index]
                x -= 1
                y += 1
        else:
            y = min(s, block_size - 1)
            x = s - y
            while y >= 0 and x < block_size:
                index += 1
                block[x, y] = array[index]
                x += 1
                y -= 1
    return block

# --- Хаффмановское кодирование ---
class HuffmanNode:
    def __init__(self, symbol=None, freq=0, left=None, right=None):
        self.symbol = symbol
        self.freq = freq
        self.left = left
        self.right = right

    def __lt__(self, other):
        return self.freq < other.freq

def build_huffman_tree(data):
    freq = {}
    for symbol in data:
        freq[symbol] = freq.get(symbol, 0) + 1
    heap = []
    for symbol, frequency in freq.items():
        node = HuffmanNode(symbol, frequency)
        heapq.heappush(heap, node)
    while len(heap) > 1:
        node1 = heapq.heappop(heap)
        node2 = heapq.heappop(heap)
        merged = HuffmanNode(None, node1.freq + node2.freq, node1, node2)
        heapq.heappush(heap, merged)
    return heap[0]

def generate_huffman_codes(root, prefix="", codebook=None):
    if codebook is None:
        codebook = {}
    if root is not None:
        if root.symbol is not None:
            codebook[root.symbol] = prefix
        generate_huffman_codes(root.left, prefix + "0", codebook)
        generate_huffman_codes(root.right, prefix + "1", codebook)
    return codebook

def huffman_encode(data, codebook):
    return "".join(codebook[symbol] for symbol in data)

def huffman_decode(bitstring, root):
    decoded = []
    node = root
    for bit in bitstring:
        node = node.left if bit == "0" else node.right
        if node.symbol is not None:
            decoded.append(node.symbol)
            node = root
    return decoded

def huffman_encode_blocks(zigzag_blocks):
    all_coeffs = []
    for block in zigzag_blocks:
        all_coeffs.extend(block)
    tree = build_huffman_tree(all_coeffs)
    codebook = generate_huffman_codes(tree)
    encoded_blocks = [huffman_encode(block, codebook) for block in zigzag_blocks]
    return encoded_blocks, tree, codebook

def huffman_decode_blocks(encoded_blocks, tree):
    decoded_blocks = [huffman_decode(encoded, tree) for encoded in encoded_blocks]
    return [np.array(block) for block in decoded_blocks]

def jpeg_compression_pipeline(image_path):
    """
    Реализует базовый процесс JPEG-сжатия для grayscale‑изображения:
      1. Разбиение изображения на блоки 8x8.
      2. Применение 2D DCT к каждому блоку.
      3. Квантование DCT‑коэффициентов с использованием стандартной матрицы.
      4. Применение зигзагообразной перестановки к каждому блоку.
      5. Энтропийное кодирование (Хаффмановское) всех блоков.
      6. Декодирование, обратная зигзагообразная перестановка.
      7. Деквантование.
      8. Применение обратного DCT.
      9. Сборка блоков в итоговое изображение.
    """
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return
    blocks = split_into_blocks(img)
    dct_blocks = apply_dct(blocks)
    quantized_blocks = quantize(dct_blocks, JPEG_QUANT_MATRIX)
    zigzag_blocks = [zigzag_scan(block) for block in quantized_blocks]
    encoded_blocks, huffman_tree, huffman_codebook = huffman_encode_blocks(zigzag_blocks)
    decoded_zigzag_blocks = huffman_decode_blocks(encoded_blocks, huffman_tree)
    recovered_quantized_blocks = [inverse_zigzag_scan(block) for block in decoded_zigzag_blocks]
    dequantized_blocks = dequantize(np.array(recovered_quantized_blocks), JPEG_QUANT_MATRIX)
    reconstructed_blocks = apply_idct(dequantized_blocks)
    cv2.imwrite("1.jpg", merge_blocks(reconstructed_blocks, img.shape))
    return merge_blocks(reconstructed_blocks, img.shape)

# def main():
#     """
#     Основная функция для вывода изображения до и после сжатия.
#     """
#     image_path = "Lena.jpg"
#
#     img_after_dct = jpeg_compression_pipeline(image_path)
#
#     original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
#
#     # Выводим исходное и обработанное изображения
#     plt.figure(figsize=(12, 6))
#
#     # Исходное изображение
#     plt.subplot(1, 3, 1)
#     plt.imshow(original_img, cmap='gray')
#     plt.title("Исходное изображение")
#     plt.axis('off')
#
#     # Изображение после DCT
#     plt.subplot(1, 3, 2)
#     plt.imshow(img_after_dct, cmap='gray')
#     plt.title("После DCT")
#     plt.axis('off')
#
#
#     plt.tight_layout()
#     plt.show()

if __name__ == "__main__":
    # 1. Создаём простую 8×8 матрицу (например, значения от 0 до 63)
    input_matrix = np.arange(64, dtype=np.float32).reshape(8, 8)
    print("Исходная 8×8 матрица:")
    print(input_matrix)

    # 2. Применяем DCT
    dct_matrix = apply_dct(input_matrix)
    print("\nDCT-коэффициенты:")
    print(np.round(dct_matrix, 2))

    # 3. Квантование: делим каждый коэффициент на соответствующий элемент матрицы квантования и округляем
    quantized = np.round(dct_matrix / JPEG_QUANT_MATRIX).astype(int)
    print("\nКвантованные DCT-коэффициенты:")
    print(quantized)

    # 4. Зигзагообразное сканирование: преобразуем 8×8 блок в одномерный вектор длины 64
    zigzag_vector = zigzag_scan(quantized)
    print("\nЗигзагообразное представление (вектор длины 64):")
    print(zigzag_vector.astype(int))

    # 5. Полное энтропийное кодирование: строим дерево Хаффмана и кодируем вектор
    huffman_tree = build_huffman_tree(zigzag_vector)
    codebook = generate_huffman_codes(huffman_tree)
    print("\nHuffman codebook:")
    for symbol in sorted(codebook.keys()):
        print(f"Символ {symbol}: {codebook[symbol]}")

    encoded_bitstream = huffman_encode(zigzag_vector, codebook)
    print("\nЗакодированный битовый поток:")
    print(encoded_bitstream)

    # 6. Полное энтропийное декодирование: декодируем битовый поток обратно в вектор
    decoded_vector = huffman_decode(encoded_bitstream, huffman_tree)
    decoded_vector = np.array(decoded_vector)
    print("\nДекодированный вектор (должен совпадать с оригинальным):")
    print(decoded_vector.astype(int))

    if np.array_equal(decoded_vector, zigzag_vector):
        print("\nДекодирование успешно: вектор совпадает с оригиналом.")
    else:
        print("\nОшибка: декодированный вектор отличается от оригинала.")

    # 7. Обратное зигзагообразное сканирование
    recovered_quantized = inverse_zigzag_scan(decoded_vector, block_size=8)
    print("\nВосстановленный блок после обратного зигзагообразного сканирования:")
    print(recovered_quantized)

    # 8. Деквантование: умножаем полученный блок на матрицу квантования
    dequantized = recovered_quantized * JPEG_QUANT_MATRIX
    print("\nДеквантованные DCT-коэффициенты:")
    print(np.round(dequantized, 2))

    # 9. Обратное DCT (IDCT) для восстановления исходной матрицы
    recovered_matrix = apply_idct(dequantized)
    print("\nВосстановленная матрица после IDCT:")
    print(np.round(recovered_matrix, 2))
