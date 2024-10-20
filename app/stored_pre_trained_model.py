import cv2
import os
from deepface import DeepFace
import numpy as np

def load_dataset(dataset_dir):
    """Load all images from the dataset folder and compute their embeddings."""
    dataset = []
    for root, dirs, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith(('jpg', 'jpeg', 'png')):
                img_path = os.path.join(root, file)
                try:
                    # Read and compute embedding
                    img = cv2.imread(img_path)
                    embedding = DeepFace.represent(img, model_name='Facenet')[0]['embedding']
                    dataset.append((img_path, embedding))
                except Exception as e:
                    print(f"Error processing {img_path}: {e}")
    return dataset

def match_face(captured_embedding, dataset, threshold=0.6):
    """Compare the captured face embedding with the dataset and find the best match."""
    best_match = None
    min_distance = float('inf')

    for img_path, stored_embedding in dataset:
        distance = np.linalg.norm(np.array(stored_embedding) - np.array(captured_embedding))
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
            captured_embedding = DeepFace.represent(captured_face, model_name='Facenet')[0]['embedding']
            best_match, distance = match_face(captured_embedding, dataset)

            if best_match:
                print(f"Match found! Best match: {best_match} with distance: {distance:.4f}")
            else:
                print("No match found. Authentication failed.")
        except Exception as e:
            print(f"Error during face matching: {e}")

def main():
    dataset_dir = "dataset"  # Replace with your dataset folder path
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
