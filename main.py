import os
import numpy as np
import cv2
import matplotlib.pyplot as plt
from Embedding_extraction import embed_watermark, extract_watermark
from Constructing_robust_features import generate_random_bijection
from quality import mse, psnr, rmse, ssim, robustness
from Review_of_JPEG_compression import jpeg_compression_pipeline


def get_cell_size_input():
    """
    Запрашивает у пользователя размер ячейки (cell_height, cell_width) и проверяет ввод.
    """
    print("\nДоступные размеры ячеек: 16×8, 8×8, 8×4, 4×4, 4×2.")
    while True:
        try:
            cell_height, cell_width = map(int, input("Введите размер ячейки (cell_height cell_width): ").split())
            if (cell_height == 16 and cell_width == 8) or \
                    (cell_height == 8 and cell_width == 8) or \
                    (cell_height == 8 and cell_width == 4) or \
                    (cell_height == 4 and cell_width == 4) or \
                    (cell_height == 4 and cell_width == 2):
                return cell_height, cell_width
            else:
                print("\n❌ Некорректный размер ячейки. Пожалуйста, выберите один из предложенных вариантов.")
        except ValueError:
            print("\n❌ Неверный формат ввода. Пожалуйста, введите два числа через пробел.")


def get_watermark_path_and_check_size(cell_height, cell_width):
    """
    Запрашивает путь к изображению водяного знака и проверяет его размер.
    Также проверяется, что размер водяного знака соответствует нужному количеству бит для выбранного размера ячейки.
    """
    # Сопоставляем размер ячеек и количество бит ЦВЗ:
    cell_to_bits = {
        (16, 8): 32,
        (8, 8): 64,
        (8, 4): 128,
        (4, 4): 256,
        (4, 2): 512
    }

    expected_bits = cell_to_bits.get((cell_height, cell_width))
    if not expected_bits:
        print(f"\n❌ Неизвестный размер ячейки: {cell_height}x{cell_width}.")
        return None

    while True:
        watermark_path = input("\nВведите путь к изображению водяного знака: ").strip()
        if os.path.isfile(watermark_path):
            watermark_img = cv2.imread(watermark_path, cv2.IMREAD_GRAYSCALE)
            if watermark_img is not None:
                height, width = watermark_img.shape
                watermark_bits = height * width
                print(f"\nРазмер водяного знака: {height}x{width} пикселей, что соответствует {watermark_bits} бит.")
                # Проверяем соответствие размера водяного знака
                if watermark_bits == expected_bits:
                    return watermark_path, watermark_bits
                else:
                    print(
                        f"\n❌ Ошибка: водяной знак должен содержать {expected_bits} бит, но у вас {watermark_bits} бит. Пожалуйста, выберите другой водяной знак.")
            else:
                print("\n❌ Ошибка при открытии изображения. Пожалуйста, попробуйте снова.")
        else:
            print("\n❌ Файл не существует. Пожалуйста, проверьте путь.")


def get_selected_bands():
    """
    Запрашивает у пользователя полосы для DCT и проверяет корректность ввода.
    В результате создается список selected_bands, где каждый элемент
    - это 64 минус номер полосы, введенный пользователем.
    """
    while True:
        try:
            # Получаем список полос, введённых пользователем
            input_bands = list(map(int, input("\nВведите полосы (например, 11 12 13): ").split()))

            # Проверяем, что все полосы находятся в допустимом диапазоне
            if all(1 <= band <= 64 for band in input_bands):
                # Для каждой введённой полосы вычисляем 64 - band
                selected_bands = [64 - band for band in input_bands]
                print(f"Выбранные полосы: {input_bands}")
                return selected_bands, input_bands
            else:
                print("\n❌ Ошибка: все полосы должны быть в диапазоне от 1 до 64. Пожалуйста, попробуйте снова.")
        except ValueError:
            print("\n❌ Неверный формат ввода. Пожалуйста, введите числа через пробел.")


def get_key_input(cell_height, cell_width):
    """
    Запрашивает у пользователя ключ (перестановку) и проверяет корректность ввода.
    """
    total_elements = cell_height * cell_width

    while True:
        print("\nВыберите способ ввода ключа:")
        print("1. Ввести ключ вручную.")
        print("2. Загрузить ключ из файла.")
        print("3. Сгенерировать случайный ключ.")

        choice = input("\nВаш выбор (1/2/3): ").strip()

        if choice == "1":
            try:
                key = list(
                    map(int, input(f"Введите перестановку чисел от 0 до {total_elements - 1} через пробел: ").split()))
                if len(key) == total_elements and len(set(key)) == total_elements:
                    return np.array(key)
                else:
                    print(
                        f"\n❌ Ошибка: ключ должен содержать уникальные значения от 0 до {total_elements - 1}. Попробуйте снова.")
            except ValueError:
                print("\n❌ Неверный формат ввода. Попробуйте снова.")

        elif choice == "2":
            key_file = input("\nВведите путь к файлу с ключом: ").strip()
            if os.path.isfile(key_file):
                with open(key_file, "r") as f:
                    try:
                        key = list(map(int, f.read().split()))
                        if len(key) == total_elements and len(set(key)) == total_elements:
                            return np.array(key)
                        else:
                            print("\n❌ Ошибка в файле: количество чисел или повторяющиеся значения.")
                    except ValueError:
                        print("\n❌ Ошибка: файл должен содержать только числа.")
            else:
                print("\n❌ Файл не существует. Пожалуйста, попробуйте снова.")

        elif choice == "3":
            key = generate_random_bijection(cell_height, cell_width)
            return key
        else:
            print("\n❌ Некорректный выбор. Пожалуйста, выберите 1, 2 или 3.")


def embed_watermark_to_image():
    """
    Основная функция для встраивания водяного знака.
    """
    print("\nВы выбрали встраивание водяного знака.")

    # Шаг 1: Ввод пути к изображению
    image_path = input("\nВведите путь к изображению для встраивания водяного знака: ").strip()

    # Шаг 2: Ввод параметров ячеек
    cell_height, cell_width = get_cell_size_input()

    # Шаг 3: Ввод пути к водяному знаку и проверка размера
    watermark_path, watermark_size = get_watermark_path_and_check_size(cell_height, cell_width)

    # Шаг 4: Ввод полос
    selected_bands, input_bands = get_selected_bands()

    # Шаг 5: Ввод ключа
    key = get_key_input(cell_height, cell_width)

    # Шаг 6: Встраивание водяного знака
    watermarked_img, watermark_bits, T, cover_shape = embed_watermark(
        cover_path=image_path,
        watermark_path=watermark_path,
        block_size=8,
        selected_bands=selected_bands,
        input_bands=input_bands,
        cell_height=cell_height,
        cell_width=cell_width,
        key=key  # Передаём ключ
    )
    # Шаг 7: Сохранение изображения
    save_path = input(
        "\nВведите имя для сохранения изображения с водяным знаком (по умолчанию 'watermarked_image.jpg'): ").strip()

    # Если имя не указано, используем дефолтное
    if not save_path:
        save_path = "watermarked_image"

    # Проверка, что расширение .jpg присутствует
    if not save_path.endswith(".jpg"):
        save_path += ".jpg"

    cv2.imwrite(save_path, watermarked_img)
    print(f"\n✅ Изображение успешно сохранено по пути {save_path}")

    # Шаг 8: Вывод информации о ключе
    print(f"\n🔑 Был использован ключ: {key}")

    # Предложение записать ключ в файл
    save_key = input("\nХотите ли вы сохранить ключ в файл 'key.txt'? (Да/Нет): ").strip().lower()
    if save_key == 'да':
        with open("key.txt", "w") as key_file:
            key_file.write(" ".join(map(str, key)))
        print("\n✅ Ключ успешно сохранён в 'key.txt'.")
    else:
        print("\nКлюч не был сохранён.")

    # Шаг 9: Сравнение изображений
    compare_images = input(
        "\nЖелаете посмотреть сравнение исходного изображения и изображения с водяным знаком? (Да/Нет): ").strip().lower()

    if compare_images == 'да':
        # Загружаем исходное изображение для сравнения
        original_img = jpeg_compression_pipeline(image_path)
        # original_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Отображаем оба изображения рядом
        plt.figure(figsize=(12, 6))

        # Исходное изображение
        plt.subplot(1, 2, 1)
        plt.imshow(original_img, cmap='gray')
        plt.title("Исходное изображение")
        plt.axis('off')

        # Изображение с водяным знаком
        plt.subplot(1, 2, 2)
        plt.imshow(watermarked_img, cmap='gray')
        plt.title("Изображение с водяным знаком")
        plt.axis('off')

        # Показываем сравнение
        plt.tight_layout()
        plt.show()


def get_key_input_for_extraction(cell_height, cell_width):
    """
    Запрашивает у пользователя ключ (перестановку) и проверяет корректность ввода.
    """
    total_elements = cell_height * cell_width

    while True:
        print("\nВыберите способ ввода ключа:")
        print("1. Ввести ключ вручную.")
        print("2. Загрузить ключ из файла.")

        choice = input("\nВаш выбор (1/2): ").strip()

        if choice == "1":
            try:
                key = list(
                    map(int, input(f"Введите перестановку чисел от 0 до {total_elements - 1} через пробел: ").split()))
                if len(key) == total_elements and len(set(key)) == total_elements:
                    return np.array(key)
                else:
                    print(
                        f"\n❌ Ошибка: ключ должен содержать уникальные значения от 0 до {total_elements - 1}. Попробуйте снова.")
            except ValueError:
                print("\n❌ Неверный формат ввода. Попробуйте снова.")

        elif choice == "2":
            key_file = input("\nВведите путь к файлу с ключом: ").strip()
            if os.path.isfile(key_file):
                with open(key_file, "r") as f:
                    try:
                        key = list(map(int, f.read().split()))
                        if len(key) == total_elements and len(set(key)) == total_elements:
                            return np.array(key)
                        else:
                            print("\n❌ Ошибка в файле: количество чисел или повторяющиеся значения.")
                    except ValueError:
                        print("\n❌ Ошибка: файл должен содержать только числа.")
            else:
                print("\n❌ Файл не существует. Пожалуйста, попробуйте снова.")

        else:
            print("\n❌ Некорректный выбор. Пожалуйста, выберите 1 или 2.")


def extract_watermark_from_image():
    """
    Основная функция для извлечения водяного знака.
    """
    print("\nВы выбрали извлечение водяного знака.")

    # Шаг 1: Ввод пути к изображению
    watermarked_image_path = input("\nВведите путь к изображению с водяным знаком: ").strip()

    # Шаг 2: Ввод параметров ячеек
    cell_height, cell_width = get_cell_size_input()

    # Шаг 3: Ввод полос
    selected_bands, _ = get_selected_bands()

    # Шаг 4: Ввод ключа
    key = get_key_input_for_extraction(cell_height, cell_width)

    # Шаг 5: Извлечение водяного знака
    watermarked_img = cv2.imread(watermarked_image_path, cv2.IMREAD_GRAYSCALE)
    if watermarked_img is None:
        raise FileNotFoundError(f"Не удалось открыть файл {watermarked_image_path}")

    # Вызов функции для извлечения водяного знака
    extracted_bits = extract_watermark(
        watermarked_img=watermarked_img,
        cover_shape=watermarked_img.shape,
        block_size=8,
        selected_bands=selected_bands,
        cell_height=cell_height,
        cell_width=cell_width,
        key=key
    )

    # Шаг 6: Сохранение извлечённого водяного знака
    wm_height, wm_width = map(int, input("Введите высоту и ширину извлекаемого ЦВЗ (wm_height wm_width): ").split())

    # Сохранение извлечённого водяного знака
    extracted_watermark_img = (extracted_bits.reshape((wm_height, wm_width)) * 255).astype(np.uint8)

    save_path = input(
        "\nВведите имя для сохранения извлечённого водяного знака (по умолчанию 'extracted_watermark.jpg'): ").strip()

    # Если имя не указано, используем дефолтное
    if not save_path:
        save_path = "extracted_watermark"

    # Проверка, что расширение .jpg присутствует
    if not save_path.endswith(".jpg"):
        save_path += ".jpg"

    cv2.imwrite(save_path, extracted_watermark_img)
    print(f"\n✅ ЦВЗ успешно сохранен по пути {save_path}")

    show_extracted = input("\nЖелаете посмотреть извлечённый водяной знак? (Да/Нет): ").strip().lower()

    if show_extracted == "да":
        plt.imshow(extracted_watermark_img, cmap='gray')
        plt.title("Извлечённый водяной знак")
        plt.axis('off')
        plt.show()


def evaluate_quality():
    """
    Оценка эффективности встраивания: расчет MSE, PSNR, RMSE, SSIM, BER и NCC.
    """
    print("\nОценка эффективности встраивания водяного знака.")

    # Шаг 1: Ввод пути к исходному изображению
    original_img_path = input("\nВведите путь к исходному изображению: ").strip()

    # Шаг 2: Ввод пути к изображению с водяным знаком
    watermarked_img_path = input("\nВведите путь к изображению с водяным знаком: ").strip()

    # Шаг 3: Ввод пути к исходному водяному знаку
    original_watermark_path = input("\nВведите путь к исходному водяному знаку: ").strip()

    # Шаг 4: Ввод пути к извлеченному водяному знаку
    extracted_watermark_path = input("\nВведите путь к извлеченному водяному знаку: ").strip()

    # Загружаем все изображения
    # original_img = cv2.imread(original_img_path, cv2.IMREAD_GRAYSCALE)
    original_img = jpeg_compression_pipeline(original_img_path)
    watermarked_img = cv2.imread(watermarked_img_path, cv2.IMREAD_GRAYSCALE)
    original_watermark = cv2.imread(original_watermark_path, cv2.IMREAD_GRAYSCALE)
    extracted_watermark = cv2.imread(extracted_watermark_path, cv2.IMREAD_GRAYSCALE)

    # Проверяем, что все изображения загружены
    if original_img is None:
        print("\n❌ Ошибка при загрузке исходного изображения. Пожалуйста, проверьте путь.")
        return
    if watermarked_img is None:
        print("\n❌ Ошибка при загрузке изображения с водяным знаком. Пожалуйста, проверьте путь.")
        return
    if original_watermark is None:
        print("\n❌ Ошибка при загрузке исходного водяного знака. Пожалуйста, проверьте путь.")
        return
    if extracted_watermark is None:
        print("\n❌ Ошибка при загрузке извлеченного водяного знака. Пожалуйста, проверьте путь.")
        return

    # Шаг 5: Расчет качества
    # Для сравнения изображений: MSE, PSNR, RMSE, SSIM
    print("\nРассчитываем метрики качества для изображений:")

    mse_value = mse(original_img, watermarked_img)
    psnr_value = psnr(original_img, watermarked_img)
    rmse_value = rmse(original_img, watermarked_img)
    ssim_value = ssim(original_img, watermarked_img)

    print(f"MSE: {mse_value}")
    print(f"PSNR: {psnr_value}")
    print(f"RMSE: {rmse_value}")
    print(f"SSIM: {ssim_value}")

    # Для водяных знаков: BER и NCC
    print("\nРассчитываем метрики качества для водяных знаков:")

    ber_value, ncc_value = robustness(original_watermark, extracted_watermark)

    print(f"BER (битовая ошибка): {ber_value}%")
    print(f"NCC (нормализованная кросс-корреляция): {ncc_value}")


def main():
    """
    Основная функция для выбора пользователем действия (встраивание, извлечение или оценка качества).
    """
    print("\033[1;34mRobust reversible watermarking of JPEG images\033[0m")
    while True:
        print("\n=====================")
        print("Выберите действие:")
        print("1. Встраивание водяного знака.")
        print("2. Извлечение водяного знака.")
        print("3. Оценка эффективности встраивания.")
        print("4. Выход.")
        print("=====================")

        choice = input("Ваш выбор (1/2/3/4): ").strip()

        if choice == "1":
            embed_watermark_to_image()
        elif choice == "2":
            extract_watermark_from_image()
        elif choice == "3":
            evaluate_quality()
        elif choice == "4":
            print("\nВыход из программы.")
            break
        else:
            print("\n❌ Неверный выбор. Пожалуйста, выберите 1, 2, 3 или 4.")


if __name__ == "__main__":
    main()
