# **Robust Reversible Watermarking of JPEG Images**

**Robust Reversible Watermarking of JPEG Images** is a Python project implementing a robust reversible watermarking (RRW) algorithm for JPEG images. This method is based on the research paper published in *Signal Processing* (Volume **224**, 2024, Article No. 109582) by **Xingyuan Liang** and **Shijun Xiang**. The algorithm embeds a watermark that can be reliably extracted even after standard image processing operations (e.g., recompression, noise addition) and allows lossless recovery of the original image.

---

## **Table of Contents**

- **[Features](#features)**
- **[Project Structure](#project-structure)**
- **[Requirements](#requirements)**
- **[Installation](#installation)**
- **[Usage](#usage)**
  - **[Embedding a Watermark](#embedding-a-watermark)**
  - **[Extracting a Watermark](#extracting-a-watermark)**
  - **[Quality Evaluation and Experimentation](#quality-evaluation-and-experimentation)**
- **[Contact](#contact)**

---

## **Features**

- **Robust and Reversible Watermarking:**  
  The algorithm embeds a watermark robust to common attacks like JPEG recompression, WebP compression, and additive noise while allowing lossless recovery of the original image.

- **Histogram Shifting of Robust Features:**  
  The core method extracts robust features from quantized DCT coefficients and modifies them using histogram shifting with a threshold **T**.

- **Quality Metrics:**  
  Metrics like **MSE**, **PSNR**, **RMSE**, **SSIM**, and robustness metrics (**BER** and **NCC**) evaluate watermark quality.

- **Experimental Framework:**  
  An experimental module is provided for testing watermark embedding under various frequency band selections.

---

## **Project Structure**

- **Review_of_JPEG_compression.py**  
  Contains JPEG compression logic: 8×8 block division, DCT, quantization, zigzag scanning, Huffman coding, and inverse transforms.

- **Constructing_robust_features.py**  
  Functions for constructing robust features using difference statistics and random bijections.

- **Shifting_the_histogram_of_robust_features.py**  
  Implements integer transformation (histogram shifting) for embedding a watermark.

- **Embedding_extraction.py**  
  Main module for embedding and extracting the watermark.

- **quality.py**  
  Contains quality assessment functions: **MSE**, **PSNR**, **RMSE**, **SSIM**, **BER**, and **NCC**.

- **main.py**  
  An interactive command-line interface to embed/extract watermarks and assess performance.

- **experiment.py**  
  Runs experiments for embedding watermarks in multiple images and compares PSNR/SSIM values.

---

## **Requirements**

- **Python 3.x**
- **NumPy**
- **OpenCV (cv2)**
- **Matplotlib**
- **SciPy**

---

## **Installation**

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/robust-reversible-watermarking.git
   cd robust-reversible-watermarking

2. **Install the Required Packages:**
    ```bash
   pip install numpy opencv-python matplotlib scipy

---

## **Usage**
**Embedding a Watermark**

To embed a watermark:
    ```
    python main.py
    ```
1. Select **Embedding a Watermark**.
2. Provide the **cover image** path.
3. Select the **cell size** (e.g., 16×8, 8×8).
4. Choose the **watermark image**.
5. Select the **DCT bands** for embedding.
6. Provide or generate the **key**.
7. Enter a filename to save the **watermarked image**.

**Extracting a Watermark**

To extract a watermark:
    ```
    python main.py
    ```
1. Select **Extract a Watermark**.
2. Provide the **watermarked image** path.
3. Enter the original **cover image** dimensions.
4. Select the **DCT bands** used during embedding.
5. Provide the **key** used in embedding.
6. Save the extracted watermark.

**Quality Evaluation and Experimentation**

To evaluate the embedding effectiveness:
    ```
    python main.py
    ```
1. Select Evaluate Quality.
2. Provide paths for:
- The original image.
- The watermarked image.
- The original watermark.
- The extracted watermark.

Metrics calculated:
- MSE (Mean Square Error)
- PSNR (Peak Signal-to-Noise Ratio)
- RMSE (Root Mean Square Error)
- SSIM (Structural Similarity Index)
- BER (Bit Error Rate)
- NCC (Normalized Cross-Correlation)

---
## **Authors**

- **Sofia Berdus** - ssberdus@edu.hse.ru
- **Adilya Kasimova** - adalkasimova@edu.hse.ru
