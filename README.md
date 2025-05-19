# Barcode-Reader

A Python project for reading and decoding Code 11 barcodes from images using OpenCV and NumPy.

## Features

- Detects and rotates barcodes for proper alignment
- Removes non-gray pixels and various types of noise
- Handles sine wave noise using FFT-based detection
- Applies adaptive thresholding, sharpening, and morphological operations
- Decodes Code 11 barcodes from processed images
- Batch processing of images in a directory
- Includes test scripts for multiple barcode images

## Project Structure

```
Barcode-Reader/
├── main.py
├── README.md
├── images/
│   └── [barcode images]
├── versions/
│   └── [test scripts]
```

## Requirements

- Python 3.x
- OpenCV (`opencv-python`)
- NumPy
- Matplotlib

Install dependencies with:

```sh
pip install opencv-python numpy matplotlib
```

## Usage

1. Place your barcode images in the `images/` directory.
2. Run the main script:

```sh
python main.py
```

The script will process each image, display intermediate results, and print the decoded barcode.

## How it Works

- **Image Preprocessing:** Removes non-gray pixels and noise.
- **Rotation Correction:** Detects and corrects barcode orientation.
- **Noise Detection:** Uses FFT to detect sine wave noise and applies adaptive thresholding if needed.
- **Sharpening & Thresholding:** Sharpens the image and binarizes it.
- **Morphological Operations:** Applies closing and opening to clean up the barcode.
- **Barcode Decoding:** Extracts and decodes the barcode using Code 11 logic.

---

### **This project is for educational purposes.**