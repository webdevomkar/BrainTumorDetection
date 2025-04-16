import cv2
import numpy as np

# Function to add a title on top of an image with black background for visibility
def add_title(image, title):
    img_copy = image.copy()
    # Add a black rectangle as text background
    cv2.rectangle(img_copy, (0, 0), (img_copy.shape[1], 40), (0, 0, 0), -1)
    cv2.putText(img_copy, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                1, (255, 255, 255), 2, cv2.LINE_AA)
    return img_copy

# Resize to uniform size for stacking
def resize_img(img, size=(300, 300)):
    return cv2.resize(img, size)

# Load image
img = cv2.imread("Resources/Y1.jpg")
img = resize_img(img)
original = add_title(img, "Original")

# Grayscale
img1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = add_title(cv2.cvtColor(resize_img(img1), cv2.COLOR_GRAY2BGR), "Grayscale")

# Thresholding
(_, img2) = cv2.threshold(img1, 155, 255, cv2.THRESH_BINARY)
thresholded = add_title(cv2.cvtColor(resize_img(img2), cv2.COLOR_GRAY2BGR), "Threshold")

# Inverse Thresholding
(_, img3) = cv2.threshold(img1, 155, 255, cv2.THRESH_BINARY_INV)
inverse_thresh = add_title(cv2.cvtColor(resize_img(img3), cv2.COLOR_GRAY2BGR), "Inverse Threshold")

# Morphological Close
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10,5))
img4 = cv2.morphologyEx(img2, cv2.MORPH_CLOSE, kernel)
morph_close = add_title(cv2.cvtColor(resize_img(img4), cv2.COLOR_GRAY2BGR), "Morph Close")

# Erode & Dilate
img5 = cv2.erode(img4, None, iterations=14)
img6 = cv2.dilate(img5, None, iterations=13)
morph_ops = add_title(cv2.cvtColor(resize_img(img6), cv2.COLOR_GRAY2BGR), "Erode + Dilate")

# Canny Edge
img7 = cv2.Canny(image=img6, threshold1=100, threshold2=200)
canny = add_title(cv2.cvtColor(resize_img(img7), cv2.COLOR_GRAY2BGR), "Canny Edge")

# Contours
contour_img = img.copy()
(cnts, _) = cv2.findContours(img7.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(contour_img, cnts, -1, (0, 0, 255), 2)
contours = add_title(resize_img(contour_img), "Contours")

# Arrange in 2 rows
row1 = np.hstack([original, gray, thresholded, inverse_thresh])
row2 = np.hstack([morph_close, morph_ops, canny, contours])
final_display = np.vstack([row1, row2])

# Show result
cv2.imshow("Brain Tumor Detection Steps", final_display)
cv2.waitKey(0)
cv2.destroyAllWindows()
