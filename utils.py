"""
utils.py

- Info/Debug functions 
- Parsing functions
"""

import pickle
import os
import shutil
import torch
import json
import logging
from datetime import datetime
from pathlib import Path
import requests


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
def save_dict_to_json(dictionary, file_path):
  with open(file_path, 'w') as file:
    json.dump(dictionary, file, indent=4)

  logging.debug("Dictionary saved as JSON file at:", file_path)

# _v2
def set_up_logging():
  """Util function to set up the logging

    Args:

    Attributes:
        
    """ 
  log_folder_path = Path("Logs")
  if not log_folder_path.exists():
    log_folder_path.mkdir(parents=True)
    print("Logs folder created.")
  else:
    print("Logs folder already exists.")

  # Get the current date and time
  current_datetime = datetime.now()
  # Extract the date, hour, and minute components as strings
  current_date = current_datetime.strftime("%Y-%m-%d")
  current_hour = current_datetime.strftime("%H")
  current_minute = current_datetime.strftime("%M")
  log_file_name = f"log_{current_date}_{current_hour}_{current_minute}.log"

  # Create the file path
  file_path = Path(log_folder_path) / log_file_name
  file_path_str = str(file_path)

  # Save the file
  try:
    logging.basicConfig(filename=file_path_str, filemode='w', level=logging.DEBUG, format='%(name)s - %(levelname)s - %(message)s')
  except AssertionError as error:
    print(f"Can't create log file >>> Error: {error}")
  
  return

def download_image(folder_path):
  """Util function to download the image (checking if the file is available)

    Args:
        file_path (str): Path of the json file
        folder_path (str): Path of the folder (to save the images)

    Attributes:
        
    """

# _v2
def download_images_from_json(file_path="multi_label_train.json", folder_path=None):
  """Util function to download the images from a json file (checking errors) in a directory.
     It will create a new "cleaned" json file contained in the location of the original one.
     The images will be donwloaded in a new directory in the folder of the json file 

    Args:
        file_path (str): Path of the json file
        folder_path (str): Path of the folder (to save the images)

    Attributes:
        
    """
  # Update the folder path if is not provided
  if folder_path is None:
    directory, images_folder = os.path.split(file_path)# Defined the name of the folder as the name of the images file
    folder_name = Path(images_folder) 
  
  logging.debug(f"Name of the images folder: {images_folder}")

  if not folder_name.exists():
    folder_name.mkdir(parents=True)
    print("Images folder created.")
  else:
    print("Images folder already exists.")

  #DEBUG Var - checking the integrity of the function
  artificial_limit, count = 5, 0

  with open(file_path) as json_file:
    data = json.load(json_file)
    cleaned_data = {} # New "cleaned" file
    for item in data.items():
      count += 1 
      image_name = item[0] # Key of the dict (name of the images)
      values = item[1] # Values on the dict (dict with url/multi-labels)
      response = requests.head(values["url"])

      if response.ok: #Checking for availability
        image_name = image_name.replace("/", "_")
        cleaned_data[image_name] = values # Update new dict of the image can be downloaded
        file_path = os.path.join(directory, folder_name, image_name)
        print(image_name)
        print(file_path)
        with open(file_path, 'wb') as file:
          file.write(requests.get(values["url"]).content)
      else:
        logging.debug(f"Imag not available from url: {values['url']}")
      
      if count == artificial_limit:
        # SAVE the cleaned_data as new file in the location of the original file
        exit(1)
  
