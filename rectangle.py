import cv2
import numpy as np


import urllib.request

# URL of the hand cascade XML file
hand_cascade_url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_hand.xml"

# Path to save the hand cascade XML file
hand_cascade_path = "hand_cascade.xml"

# Download the hand cascade XML file
urllib.request.urlretrieve(hand_cascade_url, hand_cascade_path)


# Load the hand cascade XML file
hand_cascade = cv2.CascadeClassifier('path/to/hand_cascade.xml')

# Initialize video capture from webcam
video_capture = cv2.VideoCapture(0)

# Set the expected aspect ratio of an ATM card
expected_aspect_ratio = 85 / 53  # Width / Height

while True:
    # Read frame from webcam
    ret, frame = video_capture.read()

    # Convert frame to grayscale for contour detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Detect hands in the frame
    hands = hand_cascade.detectMultiScale(blurred, 1.3, 5)

    # Process each detected hand
    for (x, y, w, h) in hands:
        # Crop the region of interest containing the hand
        hand_roi = frame[y:y+h, x:x+w]

        # Convert hand ROI to grayscale
        hand_gray = cv2.cvtColor(hand_roi, cv2.COLOR_BGR2GRAY)

        # Apply Canny edge detection on hand ROI
        hand_edges = cv2.Canny(hand_gray, 50, 150)

        # Find contours in the hand edge map
        hand_contours, _ = cv2.findContours(
            hand_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Process each contour in the hand ROI
        for contour in hand_contours:
            # Approximate the contour to a polygon
            epsilon = 0.04 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Check if the polygon has 4 vertices (rectangle)
            if len(approx) == 4:
                # Check if the contour is convex
                if cv2.isContourConvex(approx):
                    # Calculate the aspect ratio of the rectangle
                    aspect_ratio = float(w) / h

                    # Check if the aspect ratio matches the expected ATM card aspect ratio
                    if abs(aspect_ratio - expected_aspect_ratio) < 0.2:
                        # Draw a bounding box around the detected card
                        cv2.rectangle(
                            frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                        cv2.putText(frame, 'ATM Card', (x, y - 10),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Draw a rectangle around the detected hand
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, 'Hand', (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

    # Display the frame
    cv2.imshow('ATM Card and Hand Detection', frame)

    # Check for 'q' key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()
