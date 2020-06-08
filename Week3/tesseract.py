from PIL import Image
import pytesseract
import cv2
import os 

blur_threshold = 100

image = cv2.imread('./image_00.png')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Tính toán focus của ảnh (cạnh)
focus_measure = cv2.Laplacian(gray, cv2.CV_64F).var()

# Thresh: Phân tách đen trắng
gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

filename = "{}.png".format(os.getpid())
cv2.imwrite(filename, gray)

result = pytesseract.image_to_string(Image.open(filename), lang='eng+vie')
print(result)

os.remove(filename)

if focus_measure < blur_threshold:
    text = "Blurry pix"
    print("\n---" + text + "---")
    cv2.putText(gray, "{} - FM = {:.2f}".format(text, focus_measure),
    	(30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
else:
    text = "Fine pix"
    print("\n---" + text + "---")
    cv2.putText(gray, "{} - FM = {:.2f}".format(text, focus_measure),
    	(30, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

cv2.imshow("Image", image)
cv2.imshow("Gray", gray)
cv2.waitKey(0)
