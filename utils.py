"""
utils.py

- Info/Debug functions 
"""

import pickle
import os
import shutil
import torch
import json
import logging


def get_place_to_index_mapping():
    place_to_index_mapping = {}
    file1 = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "categories/places.txt"), "r")
    lines = [line.rstrip() for line in file1.readlines()]
    for idx, place in enumerate(lines):
        place_to_index_mapping[place] = idx
    file1.close()
    return place_to_index_mapping


def get_index_to_place_mapping():
    x = get_place_to_index_mapping()
    # https://dev.to/renegadecoder94/how-to-invert-a-dictionary-in-python-2150
    x = dict(map(reversed, x.items()))
    return x


def get_incident_to_index_mapping():
    incident_to_index_mapping = {}
    file1 = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "categories/incidents.txt"), "r")
    lines = [line.rstrip() for line in file1.readlines()]
    for idx, incident in enumerate(lines):
        incident_to_index_mapping[incident] = idx
    file1.close()
    return incident_to_index_mapping


def get_index_to_incident_mapping():
    x = get_incident_to_index_mapping()
    # https://dev.to/renegadecoder94/how-to-invert-a-dictionary-in-python-2150
    x = dict(map(reversed, x.items()))
    return x


def get_loaded_pickle_file(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save_checkpoint(state,
                    is_best,
                    session_name,
                    filename='checkpoint'):
    path = os.path.join(session_name, "{}.pth.tar".format(filename))
    best_path = os.path.join(session_name, "{}_best.pth.tar".format(filename))
    torch.save(state, path)
    if is_best:
        shutil.copyfile(path, best_path)


def get_loaded_json_file(path):
    with open(path, "r") as fp:
        return json.load(fp)

# _v2
def load_images_from_json(file_path="multi_label_train.json", folder_path=None):
  """Util function to download the images from a json file (checking errors) in a directory

    Args:
        file_path (str): Path of the json file
        folder_path (str): Path of the folder (to save the images)

    Attributes:
        
    """ 
  if folder_path is None:
        folder_path = os.getcwd()
        
    else:
        print("Variable is not None.")
  with open(file_path) as json_file:
    data = json.load(json_file)

    for row in data:
      print(type(row))
      print(row.keys())
      exit(1)
  
