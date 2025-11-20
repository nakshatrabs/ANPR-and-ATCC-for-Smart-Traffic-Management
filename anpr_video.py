import cv2 
import numpy as np 
from skimage.filters import threshold_local 
import tensorflow as tf 
from skimage import measure 
import imutils 
import os 
import pymysql
import pytesseract
import dbconnection

pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


# ---------- HELPER FUNCTIONS ----------

def sort_cont(character_contours): 
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in character_contours] 
    (character_contours, boundingBoxes) = zip(*sorted(zip(character_contours, boundingBoxes),
                                                     key=lambda b: b[1][i], reverse=False))
    return character_contours 


def segment_chars(plate_img, fixed_width): 
    
    V = cv2.split(cv2.cvtColor(plate_img, cv2.COLOR_BGR2HSV))[2] 
    thresh = cv2.adaptiveThreshold(V, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2) 
    thresh = cv2.bitwise_not(thresh) 

    plate_img = imutils.resize(plate_img, width=fixed_width) 
    thresh = imutils.resize(thresh, width=fixed_width) 
    bgr_thresh = cv2.cvtColor(thresh, cv2.COLOR_GRAY2BGR) 

    labels = measure.label(thresh, background=0) 
    charCandidates = np.zeros(thresh.shape, dtype='uint8') 

    characters = [] 
    for label in np.unique(labels): 
        if label == 0:
            continue

        labelMask = np.zeros(thresh.shape, dtype='uint8') 
        labelMask[labels == label] = 255

        cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
        cnts = cnts[1] if imutils.is_cv3() else cnts[0] 

        if len(cnts) > 0: 
            c = max(cnts, key=cv2.contourArea) 
            (boxX, boxY, boxW, boxH) = cv2.boundingRect(c) 

            aspectRatio = boxW / float(boxH) 
            solidity = cv2.contourArea(c) / float(boxW * boxH) 
            heightRatio = boxH / float(plate_img.shape[0]) 

            if aspectRatio < 1.0 and solidity > 0.15 and 0.5 < heightRatio < 0.95 and boxW > 14: 
                hull = cv2.convexHull(c) 
                cv2.drawContours(charCandidates, [hull], -1, 255, -1) 

    contours, _ = cv2.findContours(charCandidates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    
    if contours: 
        contours = sort_cont(contours) 
        addPixel = 4
        for c in contours: 
            (x, y, w, h) = cv2.boundingRect(c) 
            y = max(0, y - addPixel)
            x = max(0, x - addPixel)

            temp = bgr_thresh[y:y + h + addPixel * 2, x:x + w + addPixel * 2] 
            characters.append(temp) 
        return characters 
    
    return None


# ---------- PLATE FINDER CLASS ----------

class PlateFinder: 
    def __init__(self, minPlateArea, maxPlateArea): 
        self.min_area = minPlateArea 
        self.max_area = maxPlateArea 
        self.element_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (22, 3))

    def preprocess(self, input_img): 
        imgBlurred = cv2.GaussianBlur(input_img, (7, 7), 0) 
        gray = cv2.cvtColor(imgBlurred, cv2.COLOR_BGR2GRAY) 
        sobelx = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3) 
        ret2, threshold_img = cv2.threshold(sobelx, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) 
        morph_n_thresholded_img = threshold_img.copy()
        cv2.morphologyEx(threshold_img, cv2.MORPH_CLOSE, self.element_structure, morph_n_thresholded_img)
        return morph_n_thresholded_img 

    def extract_contours(self, after_preprocess): 
        contours, _ = cv2.findContours(after_preprocess, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        return contours 

    def clean_plate(self, plate): 
        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY) 
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 11, 2) 

        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
        if contours: 
            areas = [cv2.contourArea(c) for c in contours] 
            max_index = np.argmax(areas) 
            max_cnt = contours[max_index] 
            x, y, w, h = cv2.boundingRect(max_cnt) 

            if not self.ratioCheck(areas[max_index], plate.shape[1], plate.shape[0]): 
                return plate, False, None
            
            return plate, True, [x, y, w, h] 
        
        return plate, False, None

    def check_plate(self, input_img, contour): 
        min_rect = cv2.minAreaRect(contour) 
        
        if self.validateRatio(min_rect): 
            x, y, w, h = cv2.boundingRect(contour) 
            after_validation = input_img[y:y + h, x:x + w] 
            cleaned, plateFound, coords = self.clean_plate(after_validation)

            if plateFound: 
                chars_on_plate = self.find_characters_on_plate(cleaned)
                if chars_on_plate is not None and len(chars_on_plate) == 8:
                    x1, y1, w1, h1 = coords 
                    coords = (x1 + x, y1 + y)
                    return cleaned, chars_on_plate, coords 
        
        return None, None, None

    def find_possible_plates(self, input_img): 
        plates = [] 
        self.char_on_plate = [] 
        self.corresponding_area = [] 

        after_preprocess = self.preprocess(input_img) 
        possible_plate_contours = self.extract_contours(after_preprocess) 

        for cnt in possible_plate_contours: 
            plate, chars_on_plate, coords = self.check_plate(input_img, cnt) 
            if plate is not None: 
                plates.append(plate) 
                self.char_on_plate.append(chars_on_plate) 
                self.corresponding_area.append(coords) 

        return plates if plates else None

    def find_characters_on_plate(self, plate): 
        return segment_chars(plate, 400)

    def ratioCheck(self, area, width, height): 
        ratio = width / float(height)
        if ratio < 1: ratio = 1 / ratio
        return self.min_area < area < self.max_area and 3 < ratio < 6

    def preRatioCheck(self, area, width, height): 
        ratio = width / float(height)
        if ratio < 1: ratio = 1 / ratio
        return self.min_area < area < self.max_area and 2.5 < ratio < 7

    def validateRatio(self, rect): 
        (x, y), (width, height), rect_angle = rect 
        angle = -rect_angle if width > height else 90 + rect_angle
        if angle > 15 or width == 0 or height == 0:
            return False
        area = width * height 
        return self.preRatioCheck(area, width, height)


# ---------- OCR CLASS ----------

class OCR: 
    def __init__(self, modelFile, labelFile): 
        self.model_file = modelFile 
        self.label_file = labelFile 
        self.label = self.load_label(self.label_file) 
        self.graph = self.load_graph(self.model_file) 
        self.sess = tf.compat.v1.Session(graph=self.graph, 
                                         config=tf.compat.v1.ConfigProto()) 

    def load_graph(self, modelFile): 
        graph = tf.Graph() 
        graph_def = tf.compat.v1.GraphDef() 
        
        with open(modelFile, "rb") as f: 
            graph_def.ParseFromString(f.read()) 
        
        with graph.as_default(): 
            tf.import_graph_def(graph_def) 
        
        return graph 

    def load_label(self, labelFile): 
        lines = tf.io.gfile.GFile(labelFile).readlines() 
        return [l.rstrip() for l in lines]

    def convert_tensor(self, image, outSize): 
        image = cv2.resize(image, (outSize, outSize), interpolation=cv2.INTER_CUBIC) 
        np_img = cv2.normalize(image.astype('float'), None, -0.5, .5, cv2.NORM_MINMAX) 
        np_final = np.expand_dims(np_img, axis=0) 
        return np_final 

    def label_image(self, tensor): 
        input_layer = "import/input"
        output_layer = "import/final_result"

        inp = self.graph.get_operation_by_name(input_layer) 
        out = self.graph.get_operation_by_name(output_layer) 

        results = self.sess.run(out.outputs[0], {inp.outputs[0]: tensor}) 
        results = np.squeeze(results) 
        top = results.argsort()[-1:][::-1] 
        return self.label[top[0]] 

    def label_image_list(self, listImages, imageSizeOutput): 
        plate = "" 
        for img in listImages:
            plate += self.label_image(self.convert_tensor(img, imageSizeOutput))
        return plate, len(plate)



# ===================================================================
# ========================= ANPR MAIN FUNCTION ======================
# ===================================================================

def start_anpr(input_files):

    findPlate = PlateFinder(minPlateArea=4100, maxPlateArea=15000)

    model = OCR(
        modelFile=r"C:\Users\hp\OneDrive\Desktop\Projects\Infosys SpringBoard Internship\ANPR-and-ATCC-for-Smart-Traffic-Management-main\anpr model files\binary_128_0.50_ver3.pb",
        labelFile=r"C:\Users\hp\OneDrive\Desktop\Projects\Infosys SpringBoard Internship\ANPR-and-ATCC-for-Smart-Traffic-Management-main\anpr model files\binary_128_0.50_labels_ver2.txt"
    )

    # ------------ MAIN PROCESSING LOOP -------------
    for file_path in input_files:

        cap = cv2.VideoCapture(file_path)

        while cap.isOpened():
            ret, img = cap.read()

            if not ret:
                break

            cv2.imshow('original video', img)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            possible_plates = findPlate.find_possible_plates(img)

            if possible_plates:
                for i, p in enumerate(possible_plates):

                    chars_on_plate = findPlate.char_on_plate[i]

                    recognized_plate, _ = model.label_image_list(chars_on_plate, imageSizeOutput=128)


                    print(recognized_plate)

                    # ---------------- CLEAN DATA ---------------- #

                    # clean number plate
                    plate = recognized_plate.strip().upper()
                    plate = plate.replace(" ", "")
                    plate = plate.replace("\n", "")
                    plate = ''.join([c for c in plate if c.isalnum()])

                    # keep only filename for storing
                    video_filename = os.path.basename(file_path)

                    # ---------------- DB INSERT FIXED ---------------- #
                    try:
                        connection = dbconnection.get_connection()
                        with connection.cursor() as cursor:

                            sql = """
                                INSERT INTO vehicle_data 
                                (number_plate, video_file, detected_at)
                                VALUES (%s, %s, NOW())
                            """
                            cursor.execute(sql, (plate, video_filename))
                            connection.commit()

                            print("DB Saved:", plate, video_filename)

                    except Exception as e:
                        print("DB Error:", e)

                    cv2.imshow('plate', p)

                    if cv2.waitKey(25) & 0xFF == ord('q'):
                        break

        cap.release()

    cv2.destroyAllWindows()


# RUN DIRECTLY
if __name__ == "__main__":
    video_list = [
        r"C:\Users\hp\OneDrive\Desktop\Projects\Infosys SpringBoard Internship\ANPR-and-ATCC-for-Smart-Traffic-Management-main\sample detection videos\anpr sample.mp4"
    ]
    start_anpr(video_list)
