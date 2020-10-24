# text recognition
import cv2
import numpy as np
import pytesseract

# configurations
config = ('-l eng --oem 3')
pytesseract.pytesseract.tesseract_cmd = "C:/Program Files/Tesseract-OCR/tesseract.exe"


def ocr(im):
    # pytessercat
    text = pytesseract.image_to_string(im, config=config)
    # print text
    text = text.split('\n')
    return text
