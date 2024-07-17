import cv2
import pytesseract
import numpy as np

# Set the path for Tesseract if it's not added to PATH
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_lines_and_text(image_path):
    # Read the image
    image = cv2.imread(image_path)
    output_image = image.copy()

    # Use Canny edge detection
    edges = cv2.Canny(image, 50, 150)

    # Find lines using Hough Transform
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10)

    # Prepare a dictionary to store results
    result = {}

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # Extract text at the ends of the lines
            start_text = pytesseract.image_to_string(image[y1:y1 + 20, x1:x1 + 100]).strip()
            end_text = pytesseract.image_to_string(image[y2:y2 + 20, x2 - 100:x2]).strip()

            # Draw the line
            cv2.line(output_image, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw rectangles around the text areas
            cv2.rectangle(output_image, (x1, y1), (x1 + 100, y1 + 20), (255, 0, 0), 1)
            cv2.rectangle(output_image, (x2 - 100, y2), (x2, y2 + 20), (255, 0, 0), 1)

            # Store in dictionary as key-value pair
            if start_text and end_text:
                result[start_text] = end_text

            # Put text on the image
            cv2.putText(output_image, start_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
            cv2.putText(output_image, end_text, (x2 - 100, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)

    # Save or display the output image
    cv2.imshow('Detected Lines and Text', output_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return result

# Example usage
image_path = 'path/to/your/image.jpg'
text_pairs = extract_lines_and_text(image_path)

for key, value in text_pairs.items():
    print(f"{key}: {value}")
