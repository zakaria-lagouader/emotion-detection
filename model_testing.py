import cv2
import pickle
from skimage.feature import hog

# Load the saved SVM model
def load_model(model_path):
    with open(model_path, 'rb') as model_file:
        model = pickle.load(model_file)
    return model

# FEED Lables
emotions = ["neutral", "anger", "disgust", "fear", "happy", "sadness", "surprise"]

# CK Lables
# emotions = ["anger", "disgust", "fear", "happy", "sadness", "surprise", "contempt", "neutral"]

# Function to preprocess the face image and make predictions
def predict_emotion(face_image, model):
    features = hog(face_image, orientations=7, pixels_per_cell=(8, 8),cells_per_block=(4, 4),block_norm= 'L2-Hys' ,transform_sqrt = False)
    
    features = features.reshape(1, -1)

    # Make prediction using the trained model
    emotion_prediction = model.predict(features)
    
    return emotions[int(emotion_prediction[0])]

def main():
    # Load the SVM model
    model_path = "svm_model-feed.sav"
    svm_model = load_model(model_path)

    # Initialize the face cascade
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Initialize the video capture
    cap = cv2.VideoCapture("videos/video-3.mp4")
    frame_number = (0 * 60 + 0) * 30 # frame at 0:00
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)


    while cap.isOpened():
        # Read a frame from the camera
        ret, frame = cap.read()

        # Convert the frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces in the frame
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5, minSize=(30, 30))

        # Process each detected face
        for (x, y, w, h) in faces:
            # Extract the face ROI
            face_roi = gray[y:y + h, x:x + w]

            # Resize the face to a fixed size
            resized_face = cv2.resize(face_roi, (64, 64))

            # Make emotion prediction
            emotion_label = predict_emotion(resized_face, svm_model)

            # Draw the bounding box and emotion label on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f"Emotion: {emotion_label}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Display the resulting frame
        cv2.imshow('Emotion Detection', frame)

        # Break the loop when 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the capture object and close all windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()