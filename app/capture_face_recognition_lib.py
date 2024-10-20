import cv2
import pandas as pd
import face_recognition
import numpy as np

def register_user():
    """Capture user's name and face, extract face encodings, and store them in CSV."""
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
                # Convert frame to RGB as face_recognition works on RGB images
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Extract facial encodings
                encodings = face_recognition.face_encodings(rgb_frame)

                if encodings:
                    encoding = encodings[0]
                    
                    # Store the user data in a CSV without storing the image itself
                    user_data = pd.DataFrame({
                        'name': [user_name],
                        'encoding': [encoding.tolist()]  # Convert numpy array to list
                    })
                    
                    # Append to CSV
                    user_data.to_csv('user_data.csv', mode='a', header=not pd.io.common.file_exists('f_data.csv'), index=False)
                    print(f"User '{user_name}' registered successfully!")
                else:
                    print("No face detected. Please try again.")
            except Exception as e:
                print(f"Error while registering: {e}")
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Registration cancelled.")
            break

    video_capture.release()
    cv2.destroyAllWindows()

def authenticate_user():
    """Authenticate the user by comparing real-time face with stored encodings."""
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
                # Convert frame to RGB as face_recognition works on RGB images
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Extract facial encodings for the captured frame
                captured_encodings = face_recognition.face_encodings(rgb_frame)

                if captured_encodings:
                    captured_encoding = captured_encodings[0]

                    # Load stored user data
                    user_data = pd.read_csv('user_data.csv')

                    # Compare captured encoding with stored encodings
                    for index, row in user_data.iterrows():
                        stored_encoding = np.array(eval(row['encoding']))  # Convert stored list back to numpy array

                        # Calculate Euclidean distance between encodings
                        distance = np.linalg.norm(stored_encoding - captured_encoding)

                        # Define a threshold for similarity (you can tune this value)
                        if distance < 0.6:
                            print(f"Authentication successful! Welcome, {row['name']}.")
                            break
                    else:
                        print("Authentication failed: Unknown user.")
                else:
                    print("No face detected. Please try again.")
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
