import numpy as np
import cv2
from numpy.fft import fft2, ifft2
import matplotlib.pyplot as plt

# Generate Fibonacci Q-matrix
def fibonacci_q_matrix(n):
    """Generate Fibonacci Q-matrix raised to power n."""
    Q = np.array([[1, 1], [1, 0]])
    if n == 0:
        return np.eye(2, dtype=int)
    elif n == 1:
        return Q
    else:
        result = Q.copy()
        for _ in range(1, n):
            result = np.matmul(result, Q)
        return result

# Encrypt watermark using Fibonacci Q-matrix
def encrypt_watermark(watermark, key):
    """Encrypt watermark using Fibonacci Q-matrix block-wise."""
    np.random.seed(key)
    n = np.random.randint(1, 10)
    Q_n = fibonacci_q_matrix(n)

    # Split watermark into 2x2 blocks
    h, w = watermark.shape
    h_blocks, w_blocks = h // 2, w // 2
    encrypted = np.zeros_like(watermark, dtype=np.uint8)

    for i in range(h_blocks):
        for j in range(w_blocks):
            block = watermark[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2]
            if block.shape == (2, 2):
                encrypted_block = (block @ Q_n) % 256
                encrypted[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2] = encrypted_block

    return encrypted

# Decrypt watermark using Fibonacci Q-matrix
def decrypt_watermark(encrypted, key):
    """Decrypt watermark using Fibonacci Q-matrix block-wise."""
    np.random.seed(key)
    n = np.random.randint(1, 10)
    Q_n = fibonacci_q_matrix(n)
    Q_n_inv = np.linalg.inv(Q_n).astype(int) % 256

    # Split encrypted watermark into 2x2 blocks
    h, w = encrypted.shape
    h_blocks, w_blocks = h // 2, w // 2
    decrypted = np.zeros_like(encrypted, dtype=np.uint8)

    for i in range(h_blocks):
        for j in range(w_blocks):
            block = encrypted[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2]
            if block.shape == (2, 2):
                decrypted_block = (block @ Q_n_inv) % 256
                decrypted[i * 2:(i + 1) * 2, j * 2:(j + 1) * 2] = decrypted_block

    return decrypted

# Embed watermark using DFT
def embed_watermark_dft(image, key, alpha=0.1):
    """Embed a self-generated watermark into an image using DFT."""
    # Apply DFT to the image 
    dft_image = fft2(image) #chuyển sang miền tần số sử dụng dft
    dft_real, dft_imag = np.real(dft_image), np.imag(dft_image) #dft_real: phần thực dft_image: phần ảo

    # Generate self-watermark by resizing the original image and normalizing
    watermark = cv2.resize(image, (image.shape[1] // 4, image.shape[0] // 4)) # tạo watermark bằng cách giảm kích thước ảnh gốc xuống 1/4 cả dài và rộng 
    watermark = cv2.normalize(watermark, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) #chuẩn hoá giá trị pixel về miền [0,255]

    # Encrypt the watermark
    encrypted_watermark = encrypt_watermark(watermark, key)

    # Embed watermark into DFT coefficients
    h, w = encrypted_watermark.shape # lấy kích thước ma trận đã mã hoá
    dft_real[:h, :w] += alpha * encrypted_watermark # giá trị alpha để điều chỉnh độ ảnh hưởng của watermark lên ảnh

    # Combine real and imaginary parts
    watermarked_dft = dft_real + 1j * dft_imag #Kết hợp phần thực với phần ảo
    watermarked_image = np.abs(ifft2(watermarked_dft)) # đưa về lại miền không gian
    return np.clip(watermarked_image, 0, 255).astype(np.uint8), watermark

# Extract watermark using DFT
def extract_watermark_dft(watermarked_image, original_image, key, alpha=0.1):
    """Extract watermark from a watermarked image using DFT."""
    # Apply DFT to the original and watermarked images
    dft_original = fft2(original_image)
    dft_watermarked = fft2(watermarked_image)

    # Compute the difference in the DFT coefficients
    dft_diff = (np.real(dft_watermarked) - np.real(dft_original)) / alpha #xác định sự khác biệt dữa các hệ số tần số của ảnh đã nhúng và ảnh gốc nên phải chia cho alpha

    # Decrypt the watermark
    h, w = original_image.shape[0] // 4, original_image.shape[1] // 4
    decrypted_watermark = decrypt_watermark(dft_diff[:h, :w].astype(np.uint8), key)
    return np.clip(decrypted_watermark, 0, 255).astype(np.uint8)

# Main function
if __name__ == "__main__":
    # Load cover image
    cover_image = cv2.imread(r"C:\Users\HP\Desktop\Alzheimer Prediction\XLA\lung-scan-4d65fa16d3ecda33a7e58e6884b9089f0d3179b7.jpg", cv2.IMREAD_GRAYSCALE)

    # Check if image is loaded
    if cover_image is None:
        raise FileNotFoundError("Cover image not found.")

    # Parameters
    key = 3  # Encryption key
    alpha = 0.1  # Embedding strength

    # Embed watermark
    watermarked_image, watermark = embed_watermark_dft(cover_image, key, alpha)
    # cv2.imwrite("watermarked_image.jpg", watermarked_image)

    # Extract watermark
    extracted_watermark = extract_watermark_dft(watermarked_image, cover_image, key, alpha)
    # cv2.imwrite("extracted_watermark.jpg", extracted_watermark)

    # Display images
    plt.figure(figsize=(16, 8))
    plt.subplot(1, 4, 1)
    plt.title("Generated Watermark")
    plt.imshow(watermark, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 2)
    plt.title("Original Image")
    plt.imshow(cover_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 3)
    plt.title("Watermarked Image")
    plt.imshow(watermarked_image, cmap='gray')
    plt.axis('off')

    plt.subplot(1, 4, 4)
    plt.title("Extracted Watermark")
    plt.imshow(extracted_watermark, cmap='gray')
    plt.axis('off')

    plt.tight_layout()
    plt.show()
