from deepface import DeepFace
import matplotlib.pyplot as plt
import cv2

backends = ["opencv", "ssd", "dlib", "mtcnn", "retinaface", "mediapipe"]
alignment_modes = [True, False]

# Helper Methods
def plot_face(face):
    # Check if any faces are detected
    if len(face) == 0:
        print("No faces detected.")
        return
    
    # Extract the first face
    face_image = face[0]['face']
    
    # Plot the first face
    plt.figure(figsize=(5, 5))  # Adjust size for single face
    plt.imshow(face_image)
    plt.axis('off')  # Hide axes for better visualization
    plt.tight_layout()
    plt.show()
    
def capture_image():
    # Open the webcam (device 0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None
    
    print("Press 's' to capture an image or 'q' to quit.")
    
    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame.")
            break
        
        frame = cv2.flip(frame, 1)
        
        # Show the current frame
        cv2.imshow('Webcam', frame)
        
        # Wait for user input
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):  # If 's' is pressed, save the frame
            # Convert the frame from BGR to RGB
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            print("Image captured!")
            break  # Exit the loop after capturing the image
        
        elif key == ord('q'):  # If 'q' is pressed, exit without saving
            print("Quit without capturing.")
            rgb_frame = None
            break

    # Release the webcam and close windows
    cap.release()
    cv2.destroyAllWindows()
    
    return rgb_frame

# Extract face from an image

face = DeepFace.extract_faces(
    img_path='dataset/user2/user2_1.jpg',
    detector_backend=backends[3],
    align=alignment_modes[0]
)

plot_face(face)



# Check for 2 images if they are same

result = DeepFace.verify(
  img1_path = "captured_image4.jpg",
  img2_path = "dataset/user2/user2_3.jpg",
  threshold=0.45 
)

print(result['verified'])


# Capture the image and look it up in db

rgb_image = capture_image()

cv2.imwrite('captured_image7.jpg', cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR))

dfs = DeepFace.find(
  img_path = "C:/Users/Arvind/Downloads/20241011_133203.jpg",
  db_path = "dataset",
  threshold=0.46
)

print(dfs)