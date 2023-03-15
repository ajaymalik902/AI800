import numpy
from PIL import Image, ImageDraw
from cv2 import rectangle, circle, imshow, waitKey, LINE_AA, imwrite
import numpy as np
import os

# polygon = [(x1,y1),(x2,y2),...] or [x1,y1,x2,y2,...]
# width = ?
# height = ?

# img = Image.new('RGB', (1024, 1024), (128, 128, 128))
# ImageDraw.Draw(img).ellipse((512-50,512-50,512+50,512+50), fill=(255, 255, 255), outline=(0, 0, 0))
# mask = numpy.array(img)
#
# img.show()


def create_mask(img_size, coords, thickness, paths, dest_dir=None):
    print(img_size)
    img = Image.new('L', img_size, 0)
    ImageDraw.Draw(img).ellipse((coords[0] - thickness, coords[1] - thickness, coords[0] + thickness, coords[1] + thickness), fill='white')
    path = paths.split('\\')[2].replace(".jpg", "_masks/")
    if dest_dir:
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)

        save_path = os.path.join(dest_dir, path)
    else:
        if not os.path.exists("raw_images"):
            os.mkdir("raw_images")

        save_path = "raw_images/" + path
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    img.save(save_path + str(coords[0]) + "_" + str(coords[1]) + ".png")