import os
import sys
from PIL import Image
import numpy as np
WIDTH = 128
HEIGHT = 128
SIZE = (WIDTH, HEIGHT)

input_folder = "./data"
output_folder = "./prepared_data"

if not os.path.exists(output_folder):
    os.mkdir(output_folder)

for dirname, dirnames, filenames in os.walk(input_folder):
    for num, filename in enumerate(filenames):
        img = Image.open(os.path.join(dirname, filename)
                         ).convert("RGB").resize(SIZE)
        img.save(output_folder+"/"+filename)
print("Done!")
