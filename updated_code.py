import cv2
import numpy as np

def load_and_preprocess(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    # Apply binary thresholding (invert colors for better contour detection)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    return thresh

def find_inner_contour(contours, image_name):
    # Filter contours to find the inner circle based on area and circularity
    max_area = 0
    inner_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:  # Avoid division by zero
            continue
        circularity = 4 * np.pi * (area / (perimeter * perimeter))
        print(f"Contour found in {image_name}: Area = {area}, Circularity = {circularity}")
        if circularity > 0.5 and area > max_area:  # Filter by circularity and area
            max_area = area
            inner_contour = contour
    return inner_contour

def detect_inner_opening(ideal_image, sample_image):
    # Find contours to detect the inner opening of the gear
    ideal_contours, _ = cv2.findContours(ideal_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    sample_contours, _ = cv2.findContours(sample_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    print(f"Ideal contours detected: {len(ideal_contours)}")
    print(f"Sample contours detected: {len(sample_contours)}")

    ideal_inner_contour = find_inner_contour(ideal_contours, "Ideal Image")
    sample_inner_contour = find_inner_contour(sample_contours, "Sample Image")

    if ideal_inner_contour is not None and sample_inner_contour is not None:
        ideal_inner_area = cv2.contourArea(ideal_inner_contour)
        sample_inner_area = cv2.contourArea(sample_inner_contour)

        print(f"Ideal inner area: {ideal_inner_area}")
        print(f"Sample inner area: {sample_inner_area}")

        if sample_inner_area < ideal_inner_area:
            return "Missing inner opening"
        elif sample_inner_area > ideal_inner_area:
            return "Large inner opening"
    else:
        return "Inner opening not detected"
    return None

def detect_teeth_defects(ideal_image, sample_image):
    # XOR the images to highlight differences (missing or worn-out teeth)
    difference = cv2.bitwise_xor(ideal_image, sample_image)
    contours, _ = cv2.findContours(difference, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    missing_teeth_count = 0
    worn_teeth_count = 0

    for contour in contours:
        if cv2.contourArea(contour) > 50:  # Area threshold for significant defects
            # Determine if this is a missing or worn-out tooth
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = w / h

            if aspect_ratio > 1.5:
                worn_teeth_count += 1
            else:
                missing_teeth_count += 1

    results = []
    if missing_teeth_count > 0:
        results.append(f"{missing_teeth_count} missing teeth")
    if worn_teeth_count > 0:
        results.append(f"{worn_teeth_count} worn-out teeth")

    return results, contours, difference

def display_results(image, contours, difference, results):
    # Draw contours on the original image to highlight defects
    output_image = cv2.drawContours(image.copy(), contours, -1, (0, 255, 0), 2)

    # Display the difference image and the defects highlighted
    cv2.imshow("Highlighted Defects", output_image)
    cv2.imshow("Difference Image", difference)
    print("Detected gear defects:")
    for result in results:
        print(f"- {result}")

    # Hold the windows open until a key is pressed
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def main():
    # Load the ideal gear image
    ideal_path = 'ideal.jpg'
    ideal_image = load_and_preprocess(ideal_path)

    # Load the sample gear image
    sample_path = input("Enter the path to the sample gear image: ")
    sample_image = load_and_preprocess(sample_path)

    # Detect inner opening defects
    inner_opening_result = detect_inner_opening(ideal_image, sample_image)

    # Detect teeth defects
    teeth_results, contours, difference = detect_teeth_defects(ideal_image, sample_image)

    # Combine results and determine if the gear is ideal or has defects
    results = []
    if inner_opening_result:
        results.append(inner_opening_result)
    results.extend(teeth_results)

    if not results:
        results.append("Ideal gear")

    # Display the final results
    original_sample_image = cv2.imread(sample_path)
    display_results(original_sample_image, contours, difference, results)

if __name__ == "__main__":
    main()
