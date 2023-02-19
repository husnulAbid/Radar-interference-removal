import glob
import os
from os import listdir
from os.path import isfile, join
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

import cv2
import numpy as np
from PIL import Image


def get_all_files(input_1):
    all_files = []
    path = dataset_path + "/" + input_1 + "/"
    folders = [f for f in listdir(path)]

    for folder in folders:
        if folder != ".DS_Store":
            if int(folder) >= 20220415 and int(folder) < 20220501:
                fullpath = path + folder
                files = [
                    fullpath + "/" + f
                    for f in listdir(fullpath)
                    if isfile(join(fullpath, f))
                ]
                all_files.append(files)

    flattened = [val for sublist in all_files for val in sublist]

    return flattened


def get_masked_image(image_file_1):
    cap = cv2.VideoCapture(image_file_1)
    ret, original_img = cap.read()
    # print(original_img.shape)
    # Processed image
    # file_name = image_file_1
    # original_img = cv2.imread(file_name, cv2.IMREAD_COLOR)

    # Background image
    bg_file_name = "bg_ba.png"
    bg_img = cv2.imread(bg_file_name, cv2.IMREAD_COLOR)
    # print(original_img.shape)
    # print(bg_img.shape)

    # Background removal
    subtracted_img = cv2.subtract(original_img, bg_img)

    # Gray image
    gray_image = cv2.cvtColor(subtracted_img, cv2.COLOR_BGR2GRAY)

    # Image thresholding
    _, thresh1 = cv2.threshold(gray_image, 1, 255, cv2.THRESH_BINARY)

    # Erosion to remove noises
    kernel = np.ones((4, 4), np.uint8)

    # Eroded image
    eroded_image = cv2.erode(thresh1, kernel)

    # Draw bounding boxes
    boxes = []
    conts, _ = cv2.findContours(eroded_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in conts:
        area = cv2.contourArea(cnt)
        x, y, w, h = cv2.boundingRect(cnt)
        if area > 500:
            boxes.append([x, y, w, h])

    # Add the bounding boxes to the background image
    for box in boxes:
        x, y, w, h = box[0], box[1], box[2], box[3]
        cv2.rectangle(original_img, (x, y), (x + w, y + h), (200, 0, 0), thickness=1)
        bg_img = cv2.imread(bg_file_name, cv2.IMREAD_COLOR)

    # Add the croppings of the original image to the results
    res = bg_img.copy()
    for box in boxes:
        x, y, w, h = box[0], box[1], box[2], box[3]
        roi = original_img[y + 1 : y + h, x + 1 : x + w]
        bg_img = bg_img.copy()
        res[y + 1 : y + h, x + 1 : x + w] = roi

    # Change the bottom part for the dates of the data
    res[500:, :] = original_img[500:, :]

    # Final result
    # figure(figsize=(12, 10), dpi=80)
    # cvt = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    # plt.imshow(cvt, cmap="gray")
    # plt.show()
    print("output/" + image_file_1 + "result.png")
    cv2.imwrite(
        "output/" + image_file_1.split("/")[-1].split(".")[0] + " result.png", res
    )
    # Image.fromarray(res).save("output/" + image_file_1 + "result.png")


if __name__ == "__main__":
    dataset_path = "aemet/10min"

    radar_1 = "ba"

    list_1 = sorted(get_all_files(radar_1))
    for elem in list_1:
        if elem == "aemet/10min/ba/20220415/aemet_ba_202204151630.gif":
            get_masked_image(elem)

    # counter = 0
    # for tuple_item in list_1:
    # print(tuple_item[0], tuple_item[1])
    # masked_image = get_masked_image(tuple_item[0], tuple_item[1])
