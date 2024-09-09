# Import necessary libraries
import cv2
import dlib
import time
import mediapipe as mp
from fer import FER
import openai

# Initialize Dlib's face detector and facial landmark predictor for eye tracking
face_detector = dlib.get_frontal_face_detector()
landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Initialize FER (Facial Expression Recognition) detector
emotion_detector = FER()

# Initialize MediaPipe Pose for posture detection
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# Set up OpenAI API Key (You'll need to replace 'your-api-key' with your actual key)
openai.api_key = 'your-api-key'

# Function to handle eye tracking
def eye_tracking(frame, gray_frame):
    faces = face_detector(gray_frame)
    
    for face in faces:
        landmarks = landmark_predictor(gray_frame, face)
        left_eye = (landmarks.part(36).x, landmarks.part(36).y)
        right_eye = (landmarks.part(45).x, landmarks.part(45).y)

        cv2.circle(frame, left_eye, 3, (0, 255, 0), -1)
        cv2.circle(frame, right_eye, 3, (0, 255, 0), -1)

# Function to detect facial expressions
def detect_emotion(frame):
    emotion, score = emotion_detector.top_emotion(frame)
    return emotion, score

# Function to check posture using MediaPipe Pose
def detect_posture(frame):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)
    
    if results.pose_landmarks:
        for landmark in results.pose_landmarks.landmark:
            x = int(landmark.x * frame.shape[1])
            y = int(landmark.y * frame.shape[0])
            cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    return results.pose_landmarks

# Function to generate comprehension questions using GPT-4
def generate_quiz(text):
    prompt = f"Create three quiz questions based on the following text:\n\n{text}"
    response = openai.Completion.create(
        engine="gpt-4",
        prompt=prompt,
        max_tokens=150
    )
    questions = response.choices[0].text.strip()
    return questions

# Main function for real-time monitoring
def real_time_monitoring():
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale for eye tracking
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Perform eye tracking
        eye_tracking(frame, gray_frame)
        
        # Perform facial expression detection
        emotion, score = detect_emotion(frame)
        cv2.putText(frame, f'Emotion: {emotion} ({score:.2f})', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
        
        # Perform posture detection
        detect_posture(frame)
        
        # Display the frame
        cv2.imshow('Real-Time Monitoring', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# Function to check comprehension after reading
def comprehension_check():
    reading_text = "The Earth revolves around the Sun in an elliptical orbit. This takes approximately 365.25 days."
    print("Comprehension Check: Generating quiz questions...")
    questions = generate_quiz(reading_text)
    print("Quiz Questions:\n", questions)

if __name__ == "__main__":
    print("Starting real-time attention monitoring...")
    real_time_monitoring()
    
    # After the monitoring, run comprehension check
    #comprehension_check()
