import cv2
import numpy as np

# Initialize global variables for the bounding box
x_start, y_start, x_end, y_end = -1, -1, -1, -1
drawing = False  # True if mouse is pressed

# Intrinsic camera parameters (example values)
fx = 826.1702  # Focal length in pixels
fy = 815.7698
cx = 306.2125  # Principal point x
cy = 233.2915  # Principal point y
camera_matrix = np.array([[fx, 0, cx],
                          [0, fy, cy],
                          [0, 0, 1]])

# Known distance from the camera to the object plane
Z = 300  # in mm

def draw_rectangle(event, x, y, flags, param):
    global x_start, y_start, x_end, y_end, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        x_start, y_start = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            x_end, y_end = x, y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x_end, y_end = x, y

def pixel_to_realworld(u, v, Z, camera_matrix):
    """Converts pixel coordinates to real-world coordinates."""
    inv_camera_matrix = np.linalg.inv(camera_matrix)
    uv1 = np.array([u, v, 1])
    Xc = np.dot(inv_camera_matrix, uv1) * Z
    return Xc[0], Xc[1]

def get_dimensions_from_box(x_start, y_start, x_end, y_end, Z, camera_matrix):
    """Calculates the real-world dimensions from bounding box coordinates."""
    X_start, Y_start = pixel_to_realworld(x_start, y_start, Z, camera_matrix)
    X_end, Y_end = pixel_to_realworld(x_end, y_end, Z, camera_matrix)
    width = np.abs(X_end - X_start)
    height = np.abs(Y_end - Y_start)
    return width, height

def main():
    cap = cv2.VideoCapture(1)
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", draw_rectangle)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if x_start != -1 and y_start != -1 and x_end != -1 and y_end != -1:
            cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            if x_end - x_start > 0 and y_end - y_start > 0:
                width_mm, height_mm = get_dimensions_from_box(x_start, y_start, x_end, y_end, Z, camera_matrix)
                cv2.putText(frame, f"Width: {width_mm:.2f} mm, Height: {height_mm:.2f} mm", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2)

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
