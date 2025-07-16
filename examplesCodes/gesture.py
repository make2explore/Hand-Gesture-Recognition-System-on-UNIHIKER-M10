# ---------------------------------- make2explore.com -------------------------------------------------------#
# Project           - Hand Gesture Recognition System using MediaPipe & OpenCV on UNIHIKER M10
# Created By        - info@make2explore.com
# Last Modified     - 16/07/2025 17:36:00 @admin
# Software          - Python, JupyterLab, Libraries - OpenCV and MediaPipe
# Hardware          - DFRobot UNIHIKER M10 Dev Board    
# Sensors Used      - External USB WebCam - Logitech C270
# Source Repo       - github.com/make2explore
# ===========================================================================================================#
# Hand Gesture Recognition Code - 
# This code uses MediaPipe and OpenCV to perform real-time hand gesture recognition through a webcam feed.
# It detects hand landmarks, analyzes finger positions to determine their extended or bent states 
# (with special handling for thumb orientation based on handedness), and classifies the hand pose into one of 
# six gestures. Fist. Open Hand. Pointing. Victory. Thumbs Up. or Rock,, which are displayed on a full-screen 
# along with FPS metrics

import cv2  # Import OpenCV for image processing
import mediapipe as mp  # Import MediaPipe for applying ML models
import math # Import Omath lib for image processing

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Open webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

# Set window to full-screen
cv2.namedWindow('Hand Gesture Recognition', cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty('Hand Gesture Recognition', cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Gesture recognition functions
def get_finger_states(landmarks, handedness):
    finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
    mcp_joints = [2, 5, 9, 13, 17]     # Base joints for each finger
    
    states = [False] * 5
    
    # Special handling for thumb (different for left/right hands)
    if handedness == "Right":
        states[0] = landmarks[4].x > landmarks[3].x
    else:
        states[0] = landmarks[4].x < landmarks[3].x
    
    # Check other fingers (Index to Pinky)
    for i in range(1, 5):
        states[i] = landmarks[finger_tips[i]].y < landmarks[mcp_joints[i]].y
    
    return states

def recognize_gesture(finger_states):
    thumb, index, middle, ring, pinky = finger_states
    
    # Gesture detection logic
    if all(finger_states):
        return "OPEN HAND"
    elif not any(finger_states):
        return "FIST"
    elif index and not thumb and not middle and not ring and not pinky:
        return "POINTING"
    elif index and middle and not thumb and not ring and not pinky:
        return "VICTORY"
    elif thumb and not index and not middle and not ring and not pinky:
        return "THUMBS UP"
    elif index and thumb and not middle and not ring and pinky:
        return "ROCK"
    else:
        return ""

# Initialize MediaPipe Hands
with mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as hands:
    
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Convert image to RGB
        image.flags.writeable = False
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        results = hands.process(image)

        # Convert back to BGR
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        current_gesture = ""
        
        # Process hand landmarks
        if results.multi_hand_landmarks:
            for hand_landmarks, handedness in zip(
                results.multi_hand_landmarks,
                results.multi_handedness):
                
                # Get hand type (Left/Right)
                hand_type = handedness.classification[0].label
                
                # Get landmarks
                landmarks = hand_landmarks.landmark
                
                # Detect finger states
                finger_states = get_finger_states(landmarks, hand_type)
                
                # Recognize gesture
                gesture = recognize_gesture(finger_states)
                current_gesture = gesture  # Display last detected gesture
                
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
        
        # Rotate and flip image
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
        image = cv2.flip(image, 1)
        
        # Display gesture text
        if current_gesture:
            text_size = cv2.getTextSize(current_gesture, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (image.shape[1] - text_size[0]) // 2
            cv2.putText(image, current_gesture, (text_x, 40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        
        # Display FPS
        cv2.putText(image, f"FPS: {int(cap.get(cv2.CAP_PROP_FPS))}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 1, cv2.LINE_AA)
        
        # Show final image
        cv2.imshow('Hand Gesture Recognition', image)
        
        # Exit on ESC
        if cv2.waitKey(5) & 0xFF == 27:
            break

# Release resources
cap.release()
cv2.destroyAllWindows()