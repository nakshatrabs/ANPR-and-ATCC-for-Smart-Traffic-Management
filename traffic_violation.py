import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
import pytesseract
import easyocr
import re
import mysql.connector
import pytesseract
from PIL import Image
from collections import deque
from mysql.connector import Error

# Database Connection Constants
DB_HOST = 'localhost'
DB_USER = 'root'
DB_PASSWORD = 'root'
DB_NAME = 'mydb'

# Define the license plate cascade
license_plate_cascade = cv2.CascadeClassifier(
    r"C:\Users\hp\OneDrive\Desktop\Projects\Infosys SpringBoard Internship\ANPR-and-ATCC-for-Smart-Traffic-Management-main\models\YOLO\haarcascade_russian_plate_number.xml"
)
if license_plate_cascade.empty():
    raise FileNotFoundError("Haar Cascade for license plate detection not found.")


def detect_traffic_light_color(image, rect):
    x, y, w, h = rect
    roi = image[y:y+h, x:x+w]
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red_lower = np.array([0, 120, 70])
    red_upper = np.array([10, 255, 255])

    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([30, 255, 255])

    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    yellow_mask = cv2.inRange(hsv, yellow_lower, yellow_upper)

    font = cv2.FONT_HERSHEY_TRIPLEX

    if cv2.countNonZero(red_mask) > 0:
        text_color = (0, 0, 255)
        message = "Detected Signal Status: Stop"
        color = 'red'
    elif cv2.countNonZero(yellow_mask) > 0:
        text_color = (0, 255, 255)
        message = "Detected Signal Status: Caution"
        color = 'yellow'
    else:
        text_color = (0, 255, 0)
        message = "Detected Signal Status: Go"
        color = 'green'

    cv2.putText(image, message, (15, 70), font, 1.5, text_color, 3, cv2.LINE_AA)
    cv2.putText(image, 34*'-', (10, 115), font, 1, (255, 255, 255), 2, cv2.LINE_AA)

    return image, color


class LineDetector:
    def __init__(self, num_frames_avg=10):
        self.y_start_queue = deque(maxlen=num_frames_avg)
        self.y_end_queue = deque(maxlen=num_frames_avg)

    def detect_white_line(self, frame, color,
                          slope1=0.03, intercept1=920,
                          slope2=0.03, intercept2=770,
                          slope3=-0.8, intercept3=2420):

        def get_color_code(color_name):
            return {
                'red': (0, 0, 255),
                'green': (0, 255, 0),
                'yellow': (0, 255, 255)
            }.get(color_name.lower())

        frame_org = frame.copy()

        def line1(x): return slope1 * x + intercept1
        def line2(x): return slope2 * x + intercept2
        def line3(y): return slope3 * y + intercept3

        height, width, _ = frame.shape

        mask1 = frame.copy()
        for x in range(width):
            y_line = line1(x)
            mask1[int(y_line):, x] = 0

        mask2 = mask1.copy()
        for x in range(width):
            y_line = line2(x)
            mask2[:int(y_line), x] = 0

        mask3 = mask2.copy()
        for y in range(height):
            x_line = line3(y)
            mask3[y, :int(x_line)] = 0

        gray = cv2.cvtColor(mask3, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (7, 7), 0)
        edges = cv2.Canny(blurred, 30, 100)

        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100,
                                minLineLength=160, maxLineGap=5)

        x_start = 0
        x_end = width - 1

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                slope = (y2 - y1) / (x2 - x1 + 1e-8)
                intercept = y1 - slope * x1
                y_start = int(slope * x_start + intercept)
                y_end = int(slope * x_end + intercept)
                self.y_start_queue.append(y_start)
                self.y_end_queue.append(y_end)

        avg_y_start = int(sum(self.y_start_queue) / len(self.y_start_queue)) if self.y_start_queue else 0
        avg_y_end = int(sum(self.y_end_queue) / len(self.y_end_queue)) if self.y_end_queue else 0

        line_start_ratio = 0.32
        x_start_adj = x_start + int(line_start_ratio * (x_end - x_start))
        avg_y_start_adj = avg_y_start + int(line_start_ratio * (avg_y_end - avg_y_start))

        mask = np.zeros_like(frame)
        cv2.line(mask, (x_start_adj, avg_y_start_adj),
                 (x_end, avg_y_end), (255, 255, 255), 4)

        color_code = get_color_code(color)
        if color_code == (0, 255, 0):
            channels = [1]
        elif color_code == (0, 0, 255):
            channels = [2]
        else:
            channels = [1, 2]

        for c in channels:
            frame[mask[:, :, c] == 255, c] = 255

        slope_avg = (avg_y_end - avg_y_start) / (x_end - x_start + 1e-8)
        intercept_avg = avg_y_start - slope_avg * x_start

        mask_line = np.copy(frame_org)
        for x in range(width):
            y_line = slope_avg * x + intercept_avg - 35
            mask_line[:int(y_line), x] = 0

        return frame, mask_line


def extract_license_plate(frame, mask_line):

    gray = cv2.cvtColor(mask_line, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)

    kernel = np.ones((2, 2), np.uint8)
    gray = cv2.erode(gray, kernel, iterations=1)

    non_black = cv2.findNonZero(gray)
    x, y, w, h = cv2.boundingRect(non_black)
    w = int(w * 0.7)
    cropped = gray[y:y+h, x:x+w]

    plates = license_plate_cascade.detectMultiScale(
        cropped, scaleFactor=1.07, minNeighbors=15, minSize=(20, 20)
    )

    out_images = []
    for (xp, yp, wp, hp) in plates:
        cv2.rectangle(frame, (xp + x, yp + y),
                      (xp + x + wp, yp + y + hp), (0, 255, 0), 3)
        out_images.append(cropped[yp:yp+hp, xp:xp+wp])

    return frame, out_images


def apply_ocr_to_image(license_plate_image):
    _, img = cv2.threshold(license_plate_image, 120, 255, cv2.THRESH_BINARY)

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    text = pytesseract.image_to_string(img, config='--psm 8')
    text = text.strip()

    return text if text else "No text detected"


def draw_penalized_text(frame):
    font = cv2.FONT_HERSHEY_TRIPLEX
    y = 180
    cv2.putText(frame, 'Fined license plates:', (25, y),
                font, 1, (255, 255, 255), 2)
    y += 80

    for t in penalized_texts:
        cv2.putText(frame, '-> ' + t, (40, y),
                    font, 1, (255, 255, 255), 2)
        y += 60


# ------------------------ DATABASE FIXED PART ------------------------

def create_database_and_table(host, user, password, database):
    try:
        connection = mysql.connector.connect(
            host=host,
            user=user,
            password=password
        )
        cursor = connection.cursor()

        cursor.execute(f"CREATE DATABASE IF NOT EXISTS {database}")
        cursor.execute(f"USE {database}")

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS license_plates (
                id INT AUTO_INCREMENT PRIMARY KEY,
                plate_number VARCHAR(255) NOT NULL UNIQUE,
                violation_count INT DEFAULT 1
            )
        """)

        cursor.close()
        connection.close()

    except Error as e:
        print("Database error:", e)


def update_database_with_violation(plate_number, host, user, password, database):

    # Clean plate before saving
    cleaned = plate_number.strip().upper()
    cleaned = cleaned.replace(" ", "")
    cleaned = re.sub(r"[^A-Z0-9]", "", cleaned)

    try:
        connection = mysql.connector.connect(
            host=host, user=user,
            password=password, database=database
        )
        cursor = connection.cursor()

        cursor.execute("""
            INSERT INTO license_plates (plate_number, violation_count)
            VALUES (%s, 1)
            ON DUPLICATE KEY UPDATE violation_count = violation_count + 1
        """, (cleaned,))

        connection.commit()
        cursor.close()
        connection.close()

    except Error as e:
        print("DB Update Error:", e)


def print_all_violations(host, user, password, database):
    try:
        connection = mysql.connector.connect(
            host=host, user=user,
            password=password, database=database
        )
        cursor = connection.cursor()

        cursor.execute(
            "SELECT plate_number, violation_count FROM license_plates ORDER BY violation_count DESC")
        rows = cursor.fetchall()

        print("\n" + "-"*60)
        print("All Registered Traffic Violations:\n")
        for r in rows:
            print(f"{r[0]} : {r[1]}")

        cursor.close()
        connection.close()

    except Error as e:
        print("Error:", e)


def clear_license_plates(host, user, password, database):
    try:
        connection = mysql.connector.connect(
            host=host, user=user,
            password=password, database=database
        )
        cursor = connection.cursor()
        cursor.execute("DELETE FROM license_plates")
        connection.commit()
        cursor.close()
        connection.close()

    except Error as e:
        print("Error:", e)


# ------------------- MAIN PROGRAM (UNCHANGED LOGIC) -------------------

def main(video_path):
    vid = cv2.VideoCapture(video_path)

    create_database_and_table(DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)
    clear_license_plates(DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)

    if not vid.isOpened():
        print("Error opening video file")
        return

    global penalized_texts
    penalized_texts = []

    detector = LineDetector()

    global license_plate_cascade
    license_plate_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
    )

    while vid.isOpened():
        ret, frame = vid.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1280, 720))

        rect = (1000, 50, 80, 160)
        frame, detected_color = detect_traffic_light_color(frame, rect)

        frame, mask_line = detector.detect_white_line(frame, detected_color)

        frame, license_plate_images = extract_license_plate(frame, mask_line)

        for plate_img in license_plate_images:
            text = apply_ocr_to_image(plate_img)

            cleaned = text.strip().upper()
            cleaned = cleaned.replace(" ", "")
            cleaned = re.sub(r"[^A-Z0-9]", "", cleaned)

            if cleaned and cleaned not in penalized_texts:
                penalized_texts.append(cleaned)
                print(f"\nFined license plate: {cleaned}")

                update_database_with_violation(
                    cleaned, DB_HOST, DB_USER, DB_PASSWORD, DB_NAME
                )

        draw_penalized_text(frame)

        cv2.imshow('Traffic Management System', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print_all_violations(DB_HOST, DB_USER, DB_PASSWORD, DB_NAME)

    vid.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(r"C:\Users\hp\OneDrive\Desktop\Projects\Infosys SpringBoard Internship\ANPR-and-ATCC-for-Smart-Traffic-Management-main\sample detection videos\traffic_video_modified.mp4")
