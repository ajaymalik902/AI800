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


def create_mask(img, coords, thickness):
    ImageDraw.Draw(img).ellipse(
        (coords[0] - thickness, coords[1] - thickness, coords[0] + thickness, coords[1] + thickness), fill=255,
        outline=None)
