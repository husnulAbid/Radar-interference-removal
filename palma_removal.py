import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import os
from os import listdir
from os.path import join, isfile

import time
import os
from os import system
import csv
import subprocess


def process_single_image(bg_file_name, image_file_1, path):

    cap = cv2.VideoCapture(image_file_1)
    ret, original_img = cap.read()
    # print(original_img.shape)
    # Processed image
    # file_name = image_file_1
    # original_img = cv2.imread(file_name, cv2.IMREAD_COLOR)

    # Background image
    bg_img = cv2.imread(bg_file_name, cv2.IMREAD_COLOR)
    
    #print(original_img.shape)
    #print(bg_img.shape)

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
        if area > 750:
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
    cvt = cv2.cvtColor(res, cv2.COLOR_BGR2RGB)
    # plt.imshow(cvt, cmap="gray")
    # plt.show()
    print("*" * 100)
    name_splitted = path.split(".")
    name_splitted[-1] = "png"
    name_splitted[-2] += "_result"
    final_path = ".".join(name_splitted)
    print(f"Creating {final_path}")
    #print("output/" + image_file_1 + "result.png")
    cv2.imwrite(final_path, res )
    # Image.fromarray(res).save("output/" + image_file_1 + "result.png")



def process_all_images(path, bg_image):
    all_files = []
    folders = [f for f in listdir(path)]
    #print(folders)
    for folder in folders:
        gifs = [f for f in listdir(f"{path}/{folder}")]
        system(f"mkdir pm_processed/{folder}")
        for gif in gifs:
            current_gif_path = f"{path}/{folder}/{gif}"
            current_new_path = f"pm_processed/{folder}/{gif}"
            print(current_gif_path)
            #current_img = gif_to_png(current_gif_path)
            processed = process_single_image(bg_image, current_gif_path, current_new_path)
            #cv2.imwrite(f"{current_new_path}.png", processed, current_new_path)
            
        #for subfolder in subfolders:
        
            
    #flattened = [val for sublist in all_files for val in sublist]
    #print(len(flattened))


def gif_to_png(gif):
  cap = cv2.VideoCapture(gif)
  ret, image = cap.read()
  RGB_img = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
  return image
  #Image.fromarray(RGB_img).save("cloud_ba_may.png")
  
bg_image =  "pm_bg.jpg"
process_all_images("./pm", bg_image)
exit()





#####################################





def find_is_match(input_1_file, input_2_file):
    first_ = input_1_file.split("_")[-1]
    second_ = input_2_file.split("_")[-1]

    return first_ == second_


def find_matched_tuple(input_1_files, input_2_files, input_1, input_2):
    matched_tuple_list = []

    for input_1_file in input_1_files:
        for input_2_file in input_2_files:
            is_match = find_is_match(input_1_file, input_2_file)

            if is_match:
                folder_name = input_1_file.split("_")[-1][:8]
                thistuple = (
                    dataset_path
                    + "/"
                    + input_1
                    + "/"
                    + folder_name
                    + "/"
                    + input_1_file,
                    dataset_path
                    + "/"
                    + input_2
                    + "/"
                    + folder_name
                    + "/"
                    + input_2_file,
                )
                matched_tuple_list.append(thistuple)
                break
    # print(matched_tuple_list)

    return matched_tuple_list


# def get_mathched_files_pair(input_1, input_2):
#     input_1_files = get_all_files(input_1)
#     input_2_files = get_all_files(input_2)
#     matched_list = find_matched_tuple(input_1_files, input_2_files, input_1, input_2)

#     return matched_list


def get_mathched_files_pair(input_1, input_2):
    input_1_files = get_all_files(input_1)
    input_2_files = get_all_files(input_2)
    matched_list = find_matched_tuple(input_1_files, input_2_files, input_1, input_2)

    return matched_list


def get_masked_image(image_file_1, image_file_2):
    print(image_file_1)
    print(image_file_2)
    cap = cv2.VideoCapture(image_file_1)
    ret, image = cap.read()
    cap.release()

    vaCap = cv2.VideoCapture(image_file_2)
    _, va = vaCap.read()
    vaCap.release()
    # cv2.imshow(image_file_1, image)
    # cv2.imshow(image_file_2, va)

    # waits for user to press any key
    # (this is necessary to avoid Python kernel form crashing)
    # cv2.waitKey(0)

    # closing all open windows
    # cv2.destroyAllWindows()
    if ret:
        maskCap = cv2.VideoCapture("mask.gif")
        res, maskBa = maskCap.read()
        maskVa = maskBa.copy()
        for x in range(83, 479):
            for y in range(222, 479):
                xx = 83
                yy = 222
                if (
                    np.array_equal(image[y][x], np.array([252, 148, 0]))
                    or np.array_equal(image[y][x], np.array([252, 0, 0]))
                    or np.array_equal(image[y][x], np.array([252, 252, 0]))
                ) and (
                    np.array_equal(va[y - yy][x - xx], np.array([252, 148, 0]))
                    or np.array_equal(va[y - yy][x - xx], np.array([252, 0, 0]))
                    or np.array_equal(va[y - yy][x - xx], np.array([252, 252, 0]))
                ):
                    maskBa[y][x] = [255, 0, 0]
                if (
                    np.array_equal(image[y][x], np.array([252, 148, 0]))
                    or np.array_equal(image[y][x], np.array([252, 0, 0]))
                    or np.array_equal(image[y][x], np.array([252, 252, 0]))
                ) and (
                    np.array_equal(va[y - yy][x - xx], np.array([252, 148, 0]))
                    or np.array_equal(va[y - yy][x - xx], np.array([252, 0, 0]))
                    or np.array_equal(va[y - yy][x - xx], np.array([252, 252, 0]))
                ):
                    maskVa[y - yy][x - xx] = [255, 0, 0]
    Image.fromarray(maskBa).save(image_file_1 + "mask1.png")
    Image.fromarray(maskVa).save(image_file_2 + "mask2.png")