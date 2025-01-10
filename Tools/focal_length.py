import cv2
import numpy as np

KNOWN_WIDTH_MM = 210  # Width of A4 sheet in mm

# Known distance from the camera to the A4 sheet in mm
KNOWN_DISTANCE_MM = 300  # You can adjust this based on your setup

def calculate_focal_length(measured_width_pixels, known_width_mm, known_distance_mm):
    """
    Calculate the focal length of the camera.
    """
    focal_length = (measured_width_pixels * known_distance_mm) / known_width_mm
    return focal_length

def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    print("Position the A4 sheet clearly in front of the camera.")
    print("Press 's' to calculate the focal length.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        cv2.putText(frame, "Press 's' to calculate focal length", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Webcam", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('s'):
            print("Select the A4 sheet in the image.")
            r = cv2.selectROI("Webcam", frame, showCrosshair=True, fromCenter=False)
            x, y, w, h = r

            focal_length = calculate_focal_length(w, KNOWN_WIDTH_MM, KNOWN_DISTANCE_MM)
            print(f"Calculated Focal Length: {focal_length:.2f} mm")
            break

        elif key == ord('q'):
            print("Exiting...")
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
