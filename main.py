import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def rotate_barcode(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    # Calculate gradients using Sobel
    sobel_x = cv2.Sobel(blurred_image, cv2.CV_64F, 1, 0, ksize=3)  # Gradient in x direction
    sobel_y = cv2.Sobel(blurred_image, cv2.CV_64F, 0, 1, ksize=3)  # Gradient in y direction

    # Combine gradients to get edge magnitude
    sobel_magnitude = cv2.magnitude(sobel_x, sobel_y)
    sobel_magnitude = np.uint8(np.clip(sobel_magnitude, 0, 255))  # Convert to 8-bit image

    # Threshold to create a binary edge map
    _, edges = cv2.threshold(sobel_magnitude, 50, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Guard clause if there are no contours found
    if not contours:
        return image
     
    largest_contour = max(contours, key=cv2.contourArea)

    rect = cv2.minAreaRect(largest_contour)

    # Get the angle of rotation of the bounding box
    angle = rect[-1]
    
    # Adjust the angle for proper orientation
    if angle < -45:
        angle += 90  

    (height, width) = image.shape[:2]
    center = (width // 2, height // 2)
    
    # Calculate the rotation matrix
    affine_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    
    # Get the new bounding size
    cos = np.abs(affine_matrix[0, 0])
    sin = np.abs(affine_matrix[0, 1])
    width, height = int(height * sin + width * cos), int(height * cos + width * sin)

    # Apply translation as well to ensure the barcode is inside the image 
    affine_matrix[0, 2] += (width / 2) - center[0]
    affine_matrix[1, 2] += (height / 2) - center[1]

    # Rotate the image using affine 
    img_rotated = cv2.warpAffine(image, 
                                affine_matrix, 
                                (width, height), 
                                flags=cv2.INTER_LINEAR, 
                                borderMode=cv2.BORDER_CONSTANT, 
                                borderValue=(255, 255, 255))

    # If barcode is vertical rotate it to be horizontal
    if width < height:
        img_rotated = cv2.rotate(img_rotated, cv2.ROTATE_90_CLOCKWISE)
    else:
        img_rotated = img_rotated

    return img_rotated

def detect_sine_wave(image):
    # Perform FFT
    fft = np.fft.fft2(image)
    # Shift Zero frequency component to the center
    shifted = np.fft.fftshift(fft)
    
    # Calculate magnitude spectrum
    magnitude_spectrum = np.abs(shifted)
    
    # Calculate the average frequency component
    average_freq = np.mean(magnitude_spectrum)
    
    # Detect the noise by analyzing  high frequency components only
    low_freq_threshold = average_freq * 1.5  # focus on high-frequency components
    magnitude_spectrum[magnitude_spectrum < low_freq_threshold] = 0
    
    # Detect peaks having a threshold of 10% of maximum frequency
    peaks = np.where(magnitude_spectrum > np.max(magnitude_spectrum) * 0.1)
    
    # Check if there are less than 2 peaks, 
    # then there is no sine wave
    if len(peaks[0]) < 2:
        return False
    
    # Check the spacing between peaks
    peak_distances = []
    for i in range(1, len(peaks[0])):

        # calculate peaks' positions
        peak_1 = np.array([peaks[0][i], peaks[1][i]])
        peak_2 = np.array([peaks[0][i-1], peaks[1][i-1]])

        # calculate peaks' distance
        distance = np.linalg.norm(peak_1 - peak_2)
        peak_distances.append(distance)
    
    # Check if the wave is periodic, indicating a sinosoidal wave
    mean_distance = np.mean(peak_distances)
    peridicity = np.std(peak_distances) < mean_distance * 0.2
    
    return peridicity
    
def delete_non_gray(image, tolerance = 10):
    result = image.copy()
    
    # Compute the absolute difference between channels
    diff_rg = np.abs(result[:, :, 0] - result[:, :, 1])
    diff_rb = np.abs(result[:, :, 0] - result[:, :, 2])
    diff_gb = np.abs(result[:, :, 1] - result[:, :, 2])
    
    # Create a mask for gray pixels (all differences within the tolerance)
    gray_mask = (diff_rg <= tolerance) & (diff_rb <= tolerance) & (diff_gb <= tolerance)
    
    # Set non-gray pixels to white
    result[~gray_mask] = [255, 255, 255]
    
    return result

def read_barcode(stripe):
    # Convert to string of pixels in order to loop over it
    pixels = ''.join(stripe.astype(np.uint8).astype(str))

    # Need to figure out how many pixels represent a narrow bar
    narrow_bar_size = 0
    for pixel in pixels:
        if pixel == "1":
            narrow_bar_size += 1
        else:
            break
    wide_bar_size = narrow_bar_size * 2

    tolerance = 0.5
    wide_bar_size = narrow_bar_size * 2
    min_narrow = narrow_bar_size * (1 - tolerance)
    max_narrow = narrow_bar_size * (1 + tolerance)
    min_wide = wide_bar_size * (1 - tolerance)
    max_wide = wide_bar_size * (1 + tolerance)

    # Read the actual barcode
    digits = []
    pixel_index = 0
    current_digit_widths = ""
    skip_next = False

    while pixel_index < len(pixels):
        if skip_next:
            pixel_index += int(narrow_bar_size)
            skip_next = False
            continue

        count = 1
        try:
            while pixels[pixel_index] == pixels[pixel_index + 1]:
                count += 1
                pixel_index += 1
        except IndexError:
            pass
        pixel_index += 1

        if min_narrow <= count <= max_narrow:
            current_digit_widths += NARROW
        elif min_wide <= count <= max_wide:
            current_digit_widths += WIDE
        else:
            # Ignore invalid bar widths
            current_digit_widths = ""

        if current_digit_widths in code11_widths:
            digits.append(code11_widths[current_digit_widths])
            current_digit_widths = ""
            skip_next = True  # Skip the separator in the next iteration
        
    return digits

# 0 means narrow, 1 means wide
NARROW = "0"
WIDE = "1"

code11_widths = {
    "00110": "Stop/Start",
    "10001": "1",
    "01001": "2",
    "11000": "3",
    "00101": "4",
    "10100": "5",
    "01100": "6",
    "00011": "7",
    "10010": "8",
    "10000": "9",
    "00001": "0",
    "00100": "-",
}

images = {}
tests = [['Stop/Start', '1', '0', '4', '-', '1', '1', '6', '-', '1', '1', '6', 'Stop/Start'],
        ['Stop/Start', '1', '2', '3', '4', '5', '6', '7', '8', '9', '0', '-', 'Stop/Start'],
        ['Stop/Start', '1', '0', '4', '-', '1', '1', '6', '-', '1', '1', '6', 'Stop/Start'],
        ['Stop/Start', '1', '1', '2', '-', '1', '1', '5', '-', '5', '8', '-', 'Stop/Start'],
        ['Stop/Start', '-', '4', '7', '-', '4', '7', '-', '1', '2', '1', '-', 'Stop/Start'],
        ['Stop/Start', '1', '1', '1', '-', '1', '1', '7', '-', '1', '1', '6', 'Stop/Start'],
        ['Stop/Start', '-', '1', '1', '7', '-', '4', '6', '-', '9', '8', '-', 'Stop/Start'],
        ['Stop/Start', '1', '0', '1', '-', '4', '7', '-', '1', '0', '0', '-', 'Stop/Start'],
        ['Stop/Start', '1', '1', '3', '-', '1', '1', '9', '-', '5', '2', '-', 'Stop/Start'],
        ['Stop/Start', '1', '1', '9', '-', '5', '7', '-', '1', '1', '9', '-', 'Stop/Start'],
        ['Stop/Start', '1', '0', '3', '-', '1', '2', '0', '-', '9', '9', '-', 'Stop/Start'],
        ['Stop/Start', '1', '1', '3', '-', '4', '7', '-', '3', '5', '-', '3', '5', 'Stop/Start'],
]

i = 0
for file in os.listdir("images"):
    file_path = os.path.join("images", file)
    img = cv2.imread(file_path)
    images[file] = img

for name, img in images.items():
    #cv2.imshow(name, img)
    
    ################## Fix image ########################
    # Remove non-gray pixels
    img_pregray = delete_non_gray(img)

    # Convert image to gray scale (R, G, B) => Intensity 
    img_gray = cv2.cvtColor(img_pregray, cv2.COLOR_BGR2GRAY)

    # detect the image sine wave error (use the grayscale image)
    isNoised = detect_sine_wave(img_gray)
    # if there is a sine wave noise, remove it using adaptive thresholding
    if isNoised:
        img_adaptive = cv2.adaptiveThreshold(
            img_gray,
            255,                             # Max intensity value
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,  # Adaptive method
            cv2.THRESH_BINARY,               # Binary thresholding
            15,                              # Size of the local region
            2                                # Constant to fine-tune thresholding
        )
    else:
        img_adaptive = img_gray

    # Detect and rotate back a rotated image
    rotated_img = rotate_barcode(img_adaptive)

    # Compute the sharpness of the image
    sharpness_value = cv2.Laplacian(rotated_img, cv2.CV_64F).var()   # the sharper the higher 
    #print(sharpness_value)
    
    # if image is already sharp
    if sharpness_value > 250 or sharpness_value < 50:
        # Apply median filter to remove salt and pepper noise
        img_fil = cv2.medianBlur(cv2.blur(rotated_img, (1, 13)), 3)
        img_fil = cv2.medianBlur(cv2.blur(img_fil, (1, 13)), 3)
        # Threashold and separate the background and the foreground (using THRESH_OTSU flag)
        _,img_bin = cv2.threshold(img_fil, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    else:
        # Compute the intensity of the sharpening according to the sharpness value 
        intensity = np.log(sharpness_value + 1)
        
        # Construct sharpening kernel
        kernel = np.array([
            [0, -1, 0],
            [-1, 4 + intensity, -1],
            [0, -1, 0]
        ]) / intensity

        # Apply the kernel
        img_sharp = cv2.filter2D(img_gray, -1, kernel)
        _, img_bin = cv2.threshold(img_sharp, 128, 255, cv2.THRESH_BINARY)

    # Crop image
    img_inv = cv2.bitwise_not(img_bin)
    x,y,w,h = cv2.boundingRect(img_inv)
    img_crop = img_bin[y:y+h-h//4, x:x+w]

    # Close-Open
    kernel_height = img_crop.shape[0] // 5  
    kernel_width = 3
    kernel = np.ones((kernel_height, kernel_width), np.uint8)

    img_closed = cv2.morphologyEx(img_crop, cv2.MORPH_CLOSE, kernel, iterations=1)
    img_opened = cv2.morphologyEx(img_closed, cv2.MORPH_OPEN, kernel, iterations=1)

    cv2.imshow(name, img_opened)

    ################## Read Barcode #####################
    # Get the average of each column in the image
    #stripe = img_crop.mean(axis=0)
    
    # Get a stripe at the top of the barcode
    #stripe = img_crop[10, :]
    
    # Try different values of horizontal stripes (mean, median, specific stripe...)
    # 1. Get the average of each column in the image
    # 2. Get a stripe at the top of the barcode
    # 3. Get the median of each column in the image
    stripes = [img_opened.mean(axis=0), img_opened[10, :], np.median(img_opened, axis=0)]

    for stripe in stripes:
        # Set it to black or white based on its value
        stripe[stripe <= 127] = 1
        stripe[stripe > 128] = 0

        # Read that stripe of pixels
        digits = read_barcode(stripe)
        
        # If barcode was read, don't try the other stripes
        if digits:
            break

    ################### Print Test cases ################# 
    print(f"Test {i}: {'Pass' if digits == tests[i] else 'Fail'}")
    i+=1
    
    print(digits)
    cv2.waitKey(0)
