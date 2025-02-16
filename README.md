# Robust Reversible Watermarking for JPEG Images

This repository contains a Python implementation of a **Robust Reversible Watermarking (RRW)** algorithm for JPEG images, as described in the research paper:

**Robust Reversible Watermarking of JPEG Images**  
Xingyuan Liang, Shijun Xiang  
*Signal Processing, Volume 224, 2024, 109582*  
DOI: [10.1016/j.sigpro.2024.109582](https://doi.org/10.1016/j.sigpro.2024.109582)

The algorithm allows embedding a **watermark** into JPEG images while ensuring **robustness** against common image processing operations (e.g., JPEG recompression, JPEG2000 compression, WebP compression, and additive white Gaussian noise) and **reversibility** (the ability to restore the original image without loss).

## Features

- **Reversible Watermarking**: The embedded watermark can be extracted, and the original JPEG image can be restored without any loss.
- **Robustness**: The watermark is robust against common image processing operations, such as:
    - JPEG recompression
    - JPEG2000 compression
    - WebP compression
    - Additive white Gaussian noise (AWGN)
- **Frequency Band Selection**: The algorithm selects appropriate frequency bands for watermarking to minimize embedding distortion and file size expansion while maintaining high structural similarity.
- **Histogram Shifting**: The watermark is embedded by shifting the histogram of robust features constructed from quantized DCT coefficients.
- **Metadata Storage**: Metadata (e.g., threshold, selected frequency bands, and random mapping key) is saved to enable watermark extraction and image restoration.

## How It Works

The algorithm follows these steps:

### 1. **Preprocessing**:
   - Convert the **cover image** to **YCbCr** color space and extract the **Y (luminance)** channel.
   - Preprocess the **watermark image** (resize and binarize).

### 2. **JPEG Compression Simulation**:
   - Divide the **Y-channel** into 8x8 non-overlapping blocks.
   - Apply **2D Discrete Cosine Transform (DCT)** to each block.
   - Quantize the **DCT coefficients** using the standard JPEG quantization table.

### 3. **Watermark Embedding**:
   - Select **frequency bands** for watermarking.
   - Compute **robust features** using **difference statistics**.
   - Embed the watermark by **shifting the histogram** of robust features.

### 4. **Image Reconstruction**:
   - Update the quantized **DCT coefficients**.
   - **Dequantize** the coefficients and apply the **inverse DCT (IDCT)**.
   - Reconstruct the **watermarked Y-channel** and merge it with the original **Cb** and **Cr** channels.

### 5. **Watermark Extraction and Image Restoration**:
   - **Extract the watermark** from the watermarked image.
   - Restore the original JPEG image by **reversing the histogram shifting** process.

## Results

- **Watermarked Image**: The watermarked image (**watermarked.jpg**) will have the watermark embedded while maintaining high visual quality.
- **Extracted Watermark**: The extracted watermark (**extracted_watermark.jpg**) will match the original watermark.
- **Restored Image**: The restored image (**restored.jpg**) will be identical to the original cover image.

## Authors

- **Sofia Berdus** - ssberdus@edu.hse.ru
- **Adilya Kasimova** - adalkasimova@edu.hse.ru
