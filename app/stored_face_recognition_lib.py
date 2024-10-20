import cv2
import face_recognition
import os
import numpy as np

def load_dataset(dataset_dir):
    """Load all images from the dataset folder and compute their face encodings."""
    dataset = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                img_path = os.path.join(root, file)
                try:
                    # Load image and compute face encoding
                    img = face_recognition.load_image_file(img_path)
                    face_encodings = face_recognition.face_encodings(img)
                    
                    # Only store if a face encoding was found
                    if face_encodings:
                        encoding = face_encodings[0]
                        dataset.append((img_path, encoding))
                    else:
                        print(f"No face found in {img_path}")
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    return dataset

def match_face(captured_encoding, dataset, threshold=0.6):
    """Compare the captured face encoding with the dataset and find the best match."""
    best_match = None
    min_distance = float('inf')

    for img_path, stored_encoding in dataset:
        # Compute the Euclidean distance between encodings
        distance = np.linalg.norm(stored_encoding - captured_encoding)
        
        if distance < min_distance and distance < threshold:
            min_distance = distance
            best_match = img_path

    return best_match, min_distance

def capture_face():
    """Capture a face through the webcam."""
    video_capture = cv2.VideoCapture(0)
    print("Press 's' to capture your face.")
    captured_frame = None

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Failed to capture frame.")
            break

        frame = cv2.flip(frame, 1)
        cv2.imshow('Video', frame)

        if cv2.waitKey(1) & 0xFF == ord('s'):
            captured_frame = frame
            break

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()

    return captured_frame

def authenticate_user(dataset_dir):
    """Capture a face from the webcam and match it against the dataset."""
    dataset = load_dataset(dataset_dir)
    if len(dataset) == 0:
        print("No images found in the dataset.")
        return

    print("Starting authentication...")
    captured_face = capture_face()

    if captured_face is not None:
        try:
            # Convert the captured frame to RGB (face-recognition works with RGB)
            rgb_captured_face = cv2.cvtColor(captured_face, cv2.COLOR_BGR2RGB)

            # Compute face encoding for the captured frame
            captured_encodings = face_recognition.face_encodings(rgb_captured_face)
            if len(captured_encodings) > 0:
                captured_encoding = captured_encodings[0]

                # Find the best match
                best_match, distance = match_face(captured_encoding, dataset)

                if best_match:
                    print(f"Match found! Best match: {best_match} with distance: {distance:.4f}")
                else:
                    print("No match found. Authentication failed.")
            else:
                print("No face found in the captured frame.")
        except Exception as e:
            print(f"Error during face matching: {e}")

def main():
    dataset_dir = "dataset"  # Replace with the path to your dataset
    while True:
        choice = input("Authenticate (a) or quit (q): ").lower()

        if choice == 'a':
            authenticate_user(dataset_dir)
        elif choice == 'q':
            break
        else:
            print("Invalid option, try again.")

if __name__ == "__main__":
    main()
