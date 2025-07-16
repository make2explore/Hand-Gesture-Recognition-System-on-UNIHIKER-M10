# ---------------------------------- make2explore.com -------------------------------------------------------#
# Project           - Hand Gesture Recognition System using MediaPipe & OpenCV on UNIHIKER M10
# Created By        - info@make2explore.com
# Last Modified     - 16/07/2025 17:36:00 @admin
# Software          - Python, JupyterLab, Libraries - OpenCV and MediaPipe
# Hardware          - DFRobot UNIHIKER M10 Dev Board    
# Sensors Used      - External USB WebCam - Logitech C270
# Source Repo       - github.com/make2explore
# ===========================================================================================================#
# Hand Landmarks Detection Code
# This code uses OpenCV and MediaPipe to detect and track hands in real-time from a webcam feed. 
# It processes each video frame using MediaPipe’s hand tracking model, draws hand landmarks and connections 
# on the detected hands, rotates and flips the frame for display, and shows it in full-screen. 

import cv2  # Import OpenCV for image processing
import mediapipe as mp  # Import MediaPipe for applying ML models

mp_drawing = mp.solutions.drawing_utils  # Drawing utilities from MediaPipe
mp_drawing_styles = mp.solutions.drawing_styles  # Drawing styles from MediaPipe
mp_hands = mp.solutions.hands  # Import Hands model from MediaPipe

# Open webcam for real-time video stream
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)  # Set video stream width to 320
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)  # Set video stream height to 240
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Set video stream buffer size to 1

# Set window to full-screen mode
cv2.namedWindow('MediaPipe Hands', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('MediaPipe Hands', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Load Hands model
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
  while cap.isOpened():
    # Read each frame from the camera
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert image to RGB format
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Process image using Hands model
    results = hands.process(image)

    # Convert image back to BGR format
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw hand landmarks and connections if detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())

    # Rotate image 90° and flip before displaying
    image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    cv2.imshow('MediaPipe Hands', cv2.flip(image, 1))

    # Exit loop if ESC key (ASCII 27) is pressed
    if cv2.waitKey(5) & 0xFF == 27:
        break

# Release the camera
cap.release()
