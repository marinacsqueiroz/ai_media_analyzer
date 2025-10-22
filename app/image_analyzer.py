import urllib
import cv2

import numpy as np

from app.logger import LogManager

class ImageAnalyzer:
    def __init__(self, image_path):
        
        log_manager = LogManager('imageAnalyzer')
        self.logger = log_manager.get_logger()
        
        resp = urllib.request.urlopen(image_path)
        self.data = resp.read()
        
        image_array = np.asarray(bytearray(self.data), dtype=np.uint8)
        img = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        self.img = img


    def image_dimensions(self):
  
        height, width = self.img.shape[:2]
        size_kb = round(len(self.data) / 1024.0, 2)

        return {
            'width': f"{width} px",
            'height': f"{height} px",
            'size': f"{size_kb} KB"
        }

    def have_faces(self):

        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return 0
        else:
            return len(faces)

    def analyser(self):
        self.logger.info("Starting image analysis pipeline.")

        try:
            self.logger.info("Step 1: Running image dimension analysis...")
            image_dimension_result = self.image_dimensions()
            self.logger.info("Image dimension analysis completed successfully.")
        except Exception as e:
            self.logger.error(f"Error during image dimension analysis: {e}", exc_info=True)
            image_dimension_result = {"error": str(e)}

        try:
            self.logger.info("Step 2: Running face detection analysis...")
            face_result = self.have_faces()
            self.logger.info("Face detection analysis completed successfully.")
        except Exception as e:
            self.logger.error(f"Error during face detection analysis: {e}", exc_info=True)
            face_result = {"error": str(e)}

        self.logger.info("Image analysis process finished.")

        return {
            "image_analysis": {
                "image_dimension": image_dimension_result,
                "face_detected": face_result
            }
        }
