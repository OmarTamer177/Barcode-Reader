import cv2
import numpy as np

def rotateBarcode(image):
    blurred_image = cv2.GaussianBlur(image, (5, 5), 0)

    edges = cv2.Canny(blurred_image, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.int32(box)

        # Get the angle of rotation of the bounding box
        angle = rect[-1]
        
        # If the angle is too extreme, adjust it
        if angle < -45:
            angle += 90  # Correct the angle for proper orientation

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        
        # Get the new bounding box size (to prevent cropping)
        cos = np.abs(rotation_matrix[0, 0])
        sin = np.abs(rotation_matrix[0, 1])

        new_w = int(h * sin + w * cos)
        new_h = int(h * cos + w * sin)

        # Adjust the rotation matrix to take into account translation for full image retention
        rotation_matrix[0, 2] += (new_w / 2) - center[0]
        rotation_matrix[1, 2] += (new_h / 2) - center[1]

        # Rotate the image without cropping
        img_rotated = cv2.warpAffine(image, rotation_matrix, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))

        # Check if further 90-degree rotation is necessary
        if new_w < new_h:  # Check if the rotated image is taller than wide
            img_rotated_90 = cv2.rotate(img_rotated, cv2.ROTATE_90_CLOCKWISE)
        else:
            img_rotated_90 = img_rotated

    else:
        # If no contours are found, return the original image
        img_rotated_90 = image

    return img_rotated_90

def extractBarcode(image):
    img_thresh_inv = cv2.bitwise_not(image)

    x, y, w, h = cv2.boundingRect(img_thresh_inv)

    barcode = image[y:y+h-h//4, x:x+w]

    return barcode

def closeOpen(image):

    kernel_height = image.shape[0] // 5  
    kernel_width = 3
    kernel = np.ones((kernel_height, kernel_width), np.uint8)

    img_closed = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=1)

    img_opened = cv2.morphologyEx(img_closed, cv2.MORPH_OPEN, kernel, iterations=1)

    return img_opened

def histogramStretching(image):
    min = np.min(image)
    max = np.max(image)
    image = np.uint8((image - min) / (max - min) * 255)
    return image

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

# Load the image
image = cv2.imread("./images/09 - e3del el soora ya3ammm.jpg")

image = delete_non_gray(image)

image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
image = rotateBarcode(image)

cv2.imshow("rotated", image)
cv2.waitKey(0)

image = cv2.medianBlur(cv2.blur(image, (1, 13)), 3)
image = cv2.medianBlur(cv2.blur(image, (1, 13)), 3)
_,image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

cv2.imshow("blurred", image)
cv2.waitKey(0)

image = extractBarcode(image)

cv2.imshow("cropped", image)
cv2.waitKey(0)

image = closeOpen(image)

cv2.imshow("closed-opened", image)
cv2.waitKey(0)


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

# Get the average of each column in your image
mean = image.mean(axis=0)

# Set it to black or white based on its value
mean[mean <= 127] = 1
mean[mean > 128] = 0

# Convert to string of pixels in order to loop over it
pixels = ''.join(mean.astype(np.uint8).astype(str))

# Need to figure out how many pixels represent a narrow bar
narrow_bar_size = 0
for pixel in pixels:
    if pixel == "1":
        narrow_bar_size += 1
    else:
        break

tolerance = 0.3
wide_bar_size = narrow_bar_size * 2
min_narrow = narrow_bar_size * (1 - tolerance)
max_narrow = narrow_bar_size * (1 + tolerance)
min_wide = wide_bar_size * (1 - tolerance)
max_wide = wide_bar_size * (1 + tolerance)

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

print(digits)

cv2.destroyAllWindows()