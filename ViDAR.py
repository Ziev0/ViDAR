import cv2
import numpy as np
import mediapipe as mp

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

FOCAL_LENGTH = 500  # Adjust based on your camera
REAL_SHOULDER_WIDTH = 0.4  # Average human shoulder width in meters

def process_frame(image):
    """
    Processes a single frame to detect pose and calculate distance.
    :param image: Input image from the webcam.
    :return: Estimated distance and annotated image.
    """
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    results = pose.process(rgb_image)
    
    if not results.pose_landmarks:
        return None, image
    
    landmarks = results.pose_landmarks.landmark
    h, w, _ = image.shape

    # Convert keypoints to pixel coordinates
    left_shoulder = (int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w),
                     int(landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h))
    right_shoulder = (int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * w),
                      int(landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * h))
    
    cv2.circle(image, left_shoulder, 5, (0, 255, 0), -1)
    cv2.circle(image, right_shoulder, 5, (0, 255, 0), -1)
    cv2.line(image, left_shoulder, right_shoulder, (255, 0, 0), 2)
    
    # Calculate pixel length
    pixel_length = np.linalg.norm(np.array(left_shoulder) - np.array(right_shoulder))
    
    if pixel_length == 0:
        return None, image  # Avoid division by zero
    
    distance = (REAL_SHOULDER_WIDTH * FOCAL_LENGTH) / pixel_length
    
    cv2.putText(image, f"Distance: {distance:.2f}m", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    
    return distance, image

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Unable to access the webcam.")
        return
    
    print("Press 'q' to exit the application.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Unable to read from the webcam.")
            break
        
        distance, annotated_frame = process_frame(frame)

        cv2.imshow("VIDAR Output", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
