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
import orjson # Efficent parsing of python
import logging
from datetime import datetime
from pathlib import Path
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed # Parallelization 
from itertools import islice
from PIL import Image

# Use th old version from now.. No -v2 available
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


# Image loader used (check if there is a better options ..)
def image_loader(filename):
    with open(filename, 'rb') as f:
        image = Image.open(f).convert('RGB')
    return image

# _v2
def get_loaded_json_file_v2(path):
    with open(path, "rb") as f:
      return orjson.loads(f.read())

# _v2
def save_dict_to_json(dictionary, file_path):
  with open(file_path, 'w') as file:
    json.dump(dictionary, file, indent=4)

  logging.debug(f"Dictionary saved as JSON file at: {file_path}")

# _v2
def set_up_logging():
  """Util function to set up the logging
  
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
    logging.basicConfig(filename=file_path_str, filemode='w', level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
  except AssertionError as error:
    print(f"Can't create log file >>> Error: {error}")
  
  return

# _v2 - Modular function to be used either for classic or parallelized implementaion
def download_image(item, folder_path):

  """Download the image; checking for incosistent case/bad request 

    Args:
        item (tuple): (image_name, dict)
        folder_path (str): Path of the folder (to save the images)

    Return:
        In case of Downloadable image return the item (original format)

    """
  try:
      image_name = item[0] # Key of the dict (name of the images)
      values = item[1] # Values on the dict (dict with url/multi-labels)
      response = requests.get(values['url'], timeout=3)
      response.raise_for_status()
      image_name = image_name.replace("/", "_")
      file_path = os.path.join(folder_path, image_name)
      with open(file_path, 'wb') as file:
          file.write(response.content)
      print("Downloaded image:", image_name)
      return image_name, values  # Return the original data item and the renamed keys
  except requests.exceptions.RequestException as e:
      print("Error:", str(e))
      # Case of "problematic" scraping
      return None, None

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
    directory, file_name = os.path.split(file_path)# Defined the name of the folder as the name of the images file
    folder_name = Path(file_name.split(".")[0])
    folder_path = Path(directory) / Path(folder_name)
  
  logging.debug(f"Name of the images folder: {folder_name}")

  if not folder_path.exists():
    folder_path.mkdir(parents=True)
    print("Images folder created.")
  else:
    print("Images folder already exists.")

  #DEBUG Var - checking the integrity of the function
  artificial_limit, count = 5, 0

  logging.info(f"Start downloads of images in {folder_path}")
  with open(file_path) as json_file:
    data = json.load(json_file)
    cleaned_data = {} # New "cleaned" file
    for item in data.items():
      count += 1 
      key, values = download_image(item, folder_path)
      if key:
        cleaned_data[key] = values
      
      if count == artificial_limit:
        # Save the cleaned_data as new file in the location of the original file
        new_file_path = os.path.join(directory, "cleaned_" + file_name)
        save_dict_to_json(cleaned_data, new_file_path)
        return

# _v2 - Debug purpose
def take(n, iterable):
    """Return the first n items of the iterable as a list."""
    return list(islice(iterable, n))

# _v2
def download_images_from_json_parallelized(file_path="data/multi_label_train.json", folder_path=None, num_thread = 10):
  """ Parallelized download of images from a .json file.

    Args:
        file_path (str): Path of the json file
        folder_path (str): Path of the folder (to save the images)

    Attributes:
        
    """
  # Update the folder path if is not provided
  directory, file_name = os.path.split(file_path) # Take info from the original file_path

  if folder_path is None:
    folder_name = Path(file_name.split(".")[0]) # Defined the name of the folder as the name of the images file
    folder_path = Path(directory) / Path(folder_name)
  else:
    folder_path = Path(folder_path)
  
  logging.debug(f"Path of the images folder: {folder_path}")

  if not folder_path.exists():
    folder_path.mkdir(parents=True)
    print("Images folder created.")
  else:
    print("Images folder already exists.")

  logging.info(f"Start parallelized downloads of images in {folder_path}")
  with open(file_path, "rb") as f:
    data = orjson.loads(f.read())
    data = take(200, data.items()) # DEBUG purpose... just take N items
    cleaned_data = {} # New "cleaned" json file to save

    # Start parallelization
    with ThreadPoolExecutor() as executor:
      futures = [executor.submit(download_image, item, folder_path) for item in data]

      for future in as_completed(futures):
          key, values = future.result()
          if key:
              cleaned_data[key] = values

    # Save the cleaned_data as new file in the location of the original file
    new_file_path = os.path.join(directory, "cleaned_" + file_name)
    save_dict_to_json(cleaned_data, new_file_path)
    # Switch the file name to the cleaned one
    logging.info(f"Changing dataset_{file_name.split('.')[0].split('_')[-1]}_file arg (considering just the cleaning one ...)")
    
  return new_file_path

