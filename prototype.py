
import cv2
import numpy as np
import pyttsx3

# Initialize text-to-speech engine
engine = pyttsx3.init()

# Load pre-trained pedestrian detection model
hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

# Open the default camera (usually the first one)
cap = cv2.VideoCapture(0)

# Define initial parameters
roi_y = None
roi_height = None
lower_yellow = np.array([20, 100, 100])
upper_yellow = np.array([30, 255, 255])

# Main loop
while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Dynamic ROI detection
    if roi_y is None or roi_height is None:
        # Example: Use simple heuristics to detect the platform area
        edges = cv2.Canny(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), 50, 150)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 1000:  # Arbitrary threshold for contour area
                x, y, w, h = cv2.boundingRect(cnt)
                roi_y = y + h // 2
                roi_height = h // 4  # Adjust this value based on your scenario
                break

    # Dynamic color thresholding
    if roi_y is not None and roi_height is not None:
        roi = hsv[roi_y:roi_y + roi_height, :, :]
        hist = cv2.calcHist([roi], [0], None, [180], [0, 180])
        max_bin = np.argmax(hist)
        lower_yellow = np.array([max_bin - 10, 100, 100])
        upper_yellow = np.array([max_bin + 10, 255, 255])

    # Threshold the HSV image to get only yellow colors
    mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

    # Bitwise AND mask with the original image
    res_yellow = cv2.bitwise_and(frame, frame, mask=mask_yellow)

    # Detect pedestrians in the frame
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    pedestrians, _ = hog.detectMultiScale(gray)

    # Check if any pedestrian crosses the yellow line
    for (x, y, w, h) in pedestrians:
        center_y = y + h // 2
        if roi_y < center_y < roi_y + roi_height:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, "Person Crossing!", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            engine.say("Warning! Person Crossing!")
            engine.runAndWait()

    # Display the frame
    cv2.imshow('Metro Platform', frame)

    # Check for key press to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the camera and close OpenCV windows
cap.release()
cv2.destroyAllWindows()
