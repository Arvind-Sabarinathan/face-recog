import cv2
import pandas as pd
from deepface import DeepFace
import numpy as np

def register_user():
    """Capture user's name and face, extract face embeddings, and store them in CSV."""
    video_capture = cv2.VideoCapture(0)
    print("Press 's' to take a snapshot for registration.")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame.")
            break
        
        frame = cv2.flip(frame, 1)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            user_name = input("Enter the user's name: ")

            try:
                # Extract the facial embedding
                embedding = DeepFace.represent(frame, model_name='VGG-Face')[0]['embedding']

                # Store the user data in a CSV without storing the image itself
                user_data = pd.DataFrame({
                    'name': [user_name],
                    'embedding': [embedding]
                })
                
                # Append to CSV
                user_data.to_csv('f_data.csv', mode='a', header=not pd.io.common.file_exists('user_data.csv'), index=False)
                print(f"User '{user_name}' registered successfully!")
            except Exception as e:
                print(f"Error while registering: {e}")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Registration cancelled.")
            break

    video_capture.release()
    cv2.destroyAllWindows()

def authenticate_user():
    """Authenticate the user by comparing real-time face with stored embeddings."""
    video_capture = cv2.VideoCapture(0)
    print("Starting authentication...")

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame.")
            break
            
        frame = cv2.flip(frame, 1)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            try:
                # Extract embedding from the captured frame
                captured_embedding = DeepFace.represent(frame, model_name='VGG-Face')[0]['embedding']

                # Load stored user data
                user_data = pd.read_csv('f_data.csv')

                # Compare captured embedding with stored embeddings
                for index, row in user_data.iterrows():
                    stored_embedding = eval(row['embedding'])
                    distance = np.linalg.norm(np.array(stored_embedding) - np.array(captured_embedding))

                    # Define a threshold for similarity (you can tune this value)
                    if distance < 0.6:
                        print(f"Authentication successful! Welcome, {row['name']}.")
                        break
                else:
                    print("Authentication failed: Unknown user.")
            except Exception as e:
                print(f"Error during authentication: {e}")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Authentication cancelled.")
            break

    video_capture.release()
    cv2.destroyAllWindows()

def main():
    while True:
        choice = input("Register a new user (r) or authenticate (a)? Press 'q' to quit: ").lower()

        if choice == 'r':
            register_user()
        elif choice == 'a':
            authenticate_user()
        elif choice == 'q':
            break
        else:
            print("Invalid option, try again.")

if __name__ == "__main__":
    main()
