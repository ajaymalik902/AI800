from tensorflow.keras.utils import img_to_array, load_img
from PIL import Image
import numpy as np
from create_mask import create_mask


def mark_img(img_path: str, dest_dir=None):
    import os
    img = Image.open(img_path)
    copied_img = img.copy()
    image_data = np.asarray(copied_img)
    info_path = img_path.replace('ThickBloodSmears_150', 'ThickBloodSmears_150/GT_updated').replace('.jpg', '.txt')
    missing_files = []
    try:
        if os.path.exists(info_path):
            info = open(info_path.replace('.jpg', '.txt'))
    except FileNotFoundError as e:
        missing_files.append(info_path.replace('.jpg', '.txt'))
    finally:
        pass

    parasites = []
    wbcs = []
    image_size = []
    if os.path.exists(info_path):
        for line in info.readlines():
            if 'Parasite' in line.strip().split(','):
                parasites.append(line.split(','))
            elif 'White_Blood_Cell' in line.split(','):
                wbcs.append(line.split(','))
            else:
                image_size.append(line.strip().split(','))

    from cv2 import rectangle, circle, imshow, waitKey, LINE_AA, imwrite

    for para in parasites:
        color = (0, 0, 255)
        thickness = 3
        radius = 20
        start_point, end_point = get_coordinates_for_circle(para)
        # circle(image_data, (round(start_point), round(end_point)), radius, color, thickness, lineType=LINE_AA)
        create_mask(tuple(np.array(image_size[0][1:], dtype=int)), (start_point, end_point), radius, img_path, dest_dir)

    # for wbc in wbcs:
    #     color = (255, 0, 0)
    #     thickness = 2
    #     radius = 40
    #     start_point, end_point = get_coordinates_for_point(wbc)
    #     # circle(image_data, (start_point, end_point), radius, color, thickness, lineType=LINE_AA)
    #     create_mask(tuple(np.array(image_size[0][1:], dtype=int)), (start_point, end_point), radius, img_path)

    import os
    if not os.path.exists('converted_images'):
        os.mkdir("converted_images")

    if dest_dir:
        if not os.path.exists(dest_dir):
            os.mkdir(dest_dir)

        converted_img_path = img_path.replace('ThickBloodSmears_150', 'converted_images')
        img_name = converted_img_path.split('\\')[2]
        abs_path = os.path.join(dest_dir, img_name)
        imwrite(os.path.join(abs_path), image_data)
        print("Image {} marked and saved at {}".format(img_name, abs_path))



def get_coordinates_for_circle(line=list):
    temp_list = []
    for obj in line[-4:]:
        print(obj)
        temp_list.append(int(float(obj.strip('\n'))))
    temp_list = np.asarray(temp_list, dtype=int)

    return np.mean([temp_list[0], temp_list[2]]), np.mean([temp_list[1], temp_list[3]])


def get_coordinates_for_point(line=list):
    temp_list = []
    for obj in line[5:]:
        print(obj)
        temp_list.append(round(int(float(obj.strip('\n')))))
    temp_list = np.asarray(temp_list, dtype=int)

    return temp_list[0], temp_list[1]


def get_images_path():
    image_path_list = []
    import os
    if not os.path.exists('ThickBloodSmears_150'):
        return "folder ThickBloodSmears_150 does not exist."
    gen = os.walk('ThickBloodSmears_150')
    gen.__next__()
    data_list = []

    for obj in gen:
        data_list.append(obj)
    data_list = data_list[:-1]
    for p_dir, img, e_img in data_list:
        if 'GT_updated' in p_dir or 'Thumbs.db' in e_img:
            pass
        else:
            for i in e_img:
                image_path_list.append(os.path.join(p_dir, i))
    return image_path_list


def create_marked_images():
    x = get_images_path()[:1000]
    print(x)
    for images in x[:850]:
        mark_img(images, dest_dir="train_data")
    for images in x[850:]:
        mark_img(images, dest_dir="test_data")
    print("done")

if __name__ == "__main__":
    # img = Image.open('ThickBloodSmears_150/F1N_1522/20171023_110309.jpg')
    # copied_img = img.copy()
    # image_data = np.asarray(copied_img)
    #
    # info = open('ThickBloodSmears_150/GT_updated/TF1N_1522/20171023_110309.txt')
    # parasites = []
    # wbcs = []
    # for line in info.readlines():
    #     if 'Parasite' in line.strip().split(','):
    #         parasites.append(line.split(','))
    #     elif 'White_Blood_Cell' in line.split(','):
    #         wbcs.append(line.split(','))
    #
    # from cv2 import rectangle, circle, imshow, waitKey, LINE_AA, imwrite
    #
    # for para in parasites:
    #     color = (0, 0, 255)
    #     thickness = 3
    #     radius = 25
    #     start_point, end_point = get_coordinates_for_circle(para)
    #     print(round(start_point), end_point)
    #     circle(image_data, (round(start_point), round(end_point)), radius, color, thickness, lineType=LINE_AA)
    #
    # for wbc in wbcs:
    #     color = (255, 0, 0)
    #     thickness = 2
    #     radius = 40
    #     start_point, end_point = get_coordinates_for_point(wbc)
    #     circle(image_data, (start_point, end_point), radius, color, thickness, lineType=LINE_AA)
    #
    # imshow('parasites', image_data)
    # imwrite('parasites.jpg', image_data)
    # waitKey(0)
    create_marked_images()
