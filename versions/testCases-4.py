import numpy as np
import cv2
import matplotlib.pyplot as plt

img = cv2.imread("./images/00 - custom.jpg")
# img = cv2.imread("./testCaseBlur.jpg")

print(img.shape)
cv2.imshow("Image Window",img)
cv2.waitKey(0)


# convert to gray scale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img_gray.shape)

kernel = np.array([[0, -1, 0], 
                   [-1, 5, -1], 
                   [0, -1, 0]])
img_sharp = cv2.filter2D(img_gray, -1, kernel)

_, img_bin = cv2.threshold(img_sharp, 128, 255, cv2.THRESH_BINARY)
cv2.imshow("Image Binary",img_bin)
cv2.waitKey(0)

# median filter
# img_fil = cv2.medianBlur(cv2.blur(img_sharp, (1, 7)), 3)
# _,img_bin = cv2.threshold(img_fil, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
# cv2.imshow("Image Binary",img_bin)
# cv2.waitKey(0)

img_inv = cv2.bitwise_not(img_bin)
x,y,w,h = cv2.boundingRect(img_inv)

img_crop = img_bin[y:y+h-h//4, x:x+w]
# img_crop = cv2.equalizeHist(img_crop)
cv2.imshow("Image crop",img_crop)
cv2.waitKey(0)
cv2.destroyAllWindows()


# histogram 
plt.figure(figsize=(9,8))
plt.hist(img_crop.ravel(), bins=256)
plt.show()

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
mean = img_crop.mean(axis=0)

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

wide_bar_size = narrow_bar_size * 2

digits = []
pixel_index = 0
current_digit_widths = ""
skip_next = False

while pixel_index < len(pixels):

    if skip_next:
        pixel_index += narrow_bar_size
        skip_next = False
        continue

    count = 1
    try:
        while pixels[pixel_index] == pixels[pixel_index + 1]:
            count += 1
            pixel_index += 1
    except:
        pass
    pixel_index += 1

    current_digit_widths += NARROW if count == narrow_bar_size else WIDE

    if current_digit_widths in code11_widths:
        digits.append(code11_widths[current_digit_widths])
        current_digit_widths = ""
        skip_next = True  # Next iteration will be a separator, so skip it

print(digits)