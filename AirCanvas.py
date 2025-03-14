import cv2
import numpy as np
import mediapipe as mp
from collections import deque  # Import deque

# Initialize MediaPipe Hands model
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Create the canvas for drawing
paintWindow = np.zeros((480, 640, 3)) + 255

# Initialize the camera
cap = cv2.VideoCapture(0)

# Set camera resolution to a smaller size for better performance
cap.set(3, 640)
cap.set(4, 480)

# Function to update values from trackbars
def setValues(x):
    pass

# Create windows for trackbars
cv2.namedWindow("Color Detectors")
cv2.createTrackbar("Upper Hue", "Color Detectors", 153, 180, setValues)
cv2.createTrackbar("Upper Saturation", "Color Detectors", 255, 255, setValues)
cv2.createTrackbar("Upper Value", "Color Detectors", 255, 255, setValues)
cv2.createTrackbar("Lower Hue", "Color Detectors", 0, 180, setValues)
cv2.createTrackbar("Lower Saturation", "Color Detectors", 0, 255, setValues)
cv2.createTrackbar("Lower Value", "Color Detectors", 0, 255, setValues)

# Define function to show the instructions on the frame
def showInstructions(frame):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'Press "b" for Blue', (10, 30), font, 0.8, (255, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, 'Press "g" for Green', (10, 60), font, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(frame, 'Press "r" for Red', (10, 90), font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, 'Press "y" for Yellow', (10, 120), font, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, 'Press "c" to Clear Canvas', (10, 150), font, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

# List for storing points of different colors
bpoints = [deque(maxlen=512)]
gpoints = [deque(maxlen=512)]
rpoints = [deque(maxlen=512)]
ypoints = [deque(maxlen=512)]

# Set initial color index
colorIndex = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip the frame for a mirror view
    frame = cv2.flip(frame, 1)

    # Convert to RGB (MediaPipe uses RGB, not BGR)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Get hand landmarks from MediaPipe
    results = hands.process(rgb_frame)

    # Reset the frame to the drawing canvas
    img = paintWindow.copy()

    # Check if landmarks are found
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get the landmarks (specific hand joints)
            for landmark in hand_landmarks.landmark:
                h, w, c = frame.shape
                cx, cy = int(landmark.x * w), int(landmark.y * h)

                # Track the center of the index finger (landmark 8)
                if landmark == hand_landmarks.landmark[8]:
                    center = (cx, cy)

                    # Add the center point to the list for the selected color
                    if colorIndex == 0:
                        bpoints[0].appendleft(center)
                    elif colorIndex == 1:
                        gpoints[0].appendleft(center)
                    elif colorIndex == 2:
                        rpoints[0].appendleft(center)
                    elif colorIndex == 3:
                        ypoints[0].appendleft(center)

            # Draw the hand landmarks and lines between them
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Only draw the points when a hand is detected
    for i in range(len(bpoints)):
        for j in range(1, len(bpoints[i])):
            if bpoints[i][j - 1] is None or bpoints[i][j] is None:
                continue
            cv2.line(frame, bpoints[i][j - 1], bpoints[i][j], (255, 0, 0), 5)

    for i in range(len(gpoints)):
        for j in range(1, len(gpoints[i])):
            if gpoints[i][j - 1] is None or gpoints[i][j] is None:
                continue
            cv2.line(frame, gpoints[i][j - 1], gpoints[i][j], (0, 255, 0), 5)

    for i in range(len(rpoints)):
        for j in range(1, len(rpoints[i])):
            if rpoints[i][j - 1] is None or rpoints[i][j] is None:
                continue
            cv2.line(frame, rpoints[i][j - 1], rpoints[i][j], (0, 0, 255), 5)

    for i in range(len(ypoints)):
        for j in range(1, len(ypoints[i])):
            if ypoints[i][j - 1] is None or ypoints[i][j] is None:
                continue
            cv2.line(frame, ypoints[i][j - 1], ypoints[i][j], (0, 255, 255), 5)

    # Show instructions on the frame
    showInstructions(frame)

    # Display the frame
    cv2.imshow("AirCanvas", frame)

    key = cv2.waitKey(1) & 0xFF

    # Break the loop if 'q' is pressed
    if key == ord('q'):
        break

    # Simulate button clicks for color change
    if key == ord('b'):  # 'b' for blue
        colorIndex = 0
    elif key == ord('g'):  # 'g' for green
        colorIndex = 1
    elif key == ord('r'):  # 'r' for red
        colorIndex = 2
    elif key == ord('y'):  # 'y' for yellow
        colorIndex = 3
    elif key == ord('c'):  # 'c' to clear canvas
        bpoints = [deque(maxlen=512)]
        gpoints = [deque(maxlen=512)]
        rpoints = [deque(maxlen=512)]
        ypoints = [deque(maxlen=512)]

# Release the camera and close the windows
cap.release()
cv2.destroyAllWindows()
