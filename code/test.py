# -*- coding=utf-8 -*-
import pytesseract
from PIL import Image

code = pytesseract.image_to_string(Image.open('../image/id.png'))
print (code)

