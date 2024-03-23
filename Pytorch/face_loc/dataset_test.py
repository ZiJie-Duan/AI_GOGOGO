"""
this file is used to test the dataset
read csv file and show the image
lable the bounding box and landmark by cv2
"""
import os
import cv2
import csv

def read_csv_file(file_path):
    """
    read the csv file
    :param file_path: the path of the csv file
    :return: the list of the csv file
    """
    with open(file_path, 'r') as f:
        reader = csv.reader(f)
        lines = list(reader)
    return lines


