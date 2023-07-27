"""
dataset.py

- Parse the dataset images (from .json)
"""

import random
import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import torchvision.transforms as transforms
from collections import defaultdict
import copy
from torchvision.datasets import CIFAR100
import logging
from math import floor
from concurrent.futures import ThreadPoolExecutor, as_completed # Parallelization of parsing images values

from utils import (
    get_place_to_index_mapping,
    get_incident_to_index_mapping,
    get_loaded_json_file,
    get_loaded_json_file_v2,
    download_images_from_json,
    download_images_from_json_parallelized,
    image_loader, # Deprecated
    image_loader_v2
)

# Reproducibility of the experiments
np.random.seed(10)

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']

# _v2
def get_train_idx(data, split_percentage=80):
  """Return a train split (keys) of the provided dict of images

    Args:
        data (dict): Every key is a image path
        split_percentage (int): It is the percent split to be used

    Returns:
        train_keys, val_keys (list, list): List of keys (training and validation keys)
    """
  logging.debug(f"Splitting train/val keys (percentage: {split_percentage}) from the '_train.json' file")

  keys = set(data.keys()) # Set/List are both equivalent; the dict keys are unique anyway ..
  num_train = floor((split_percentage / 100) * len(keys)) 
  train_keys = random.sample(keys, num_train)
  val_keys = keys.difference(train_keys)
  return list(train_keys), list(val_keys)


def is_image_file(filename):
    """Checks if a file is an image provided some known extension.

    Args:
        filename (string): Path of the image file

    Returns:
        bool: True if the filename is considered an image
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in IMG_EXTENSIONS)

# _v2
def get_vectors_v2(data, mapping, vector_len):
  """Get the vector of labels (0/1) and weights vector (0/1) for our labeled images.

    Args:
        data (dict): 
            {
                "place_or_disater": +1/0 (int)
            }
        mapping (dict):
            {
                "place_or_disater": index (int)
            }
        vector_len (int): lenth of vector (includes "no place" or "no incident")

    Returns:
        tuple: (vector with +1/0 for labels, weight vector with +1 where we have information (negative and positive examples))
    """
  vector = np.zeros(vector_len)
  pos_labels = sum(data.values()) # Number of positive examples

  if pos_labels > 0:
    # Assume full information
    weight_vector = np.ones(vector_len)
    # Put +1 label just in the pos examples found
    for key, values in data.items():
      if values == 1:
        class_index = mapping[key]
        vector[class_index] = 1
  else:
    # The vector just have negative examples
    weight_vector = np.zeros(vector_len) # Put one just where we have the information
    for key, values in data.items():
      class_index = mapping[key]
      weight_vector[class_index] = 1

  return vector, weight_vector


# Old mapping for class/weight for training
def get_vectors(data, to_index_mapping, vector_len):
    """Get the vector of labels and weight vector for our labeled iamges.

    Args:
        data (dict): 
            {
                "place_or_disater": +1/-1
            }
        to_index_mapping (dict):
            {
                "place_or_disater": index (int)
            }
        vector_len (int): lenth of vector (includes "no place" or "no incident")

    Returns:
        tuple: (vector with +1/0/-1 for labels, weight vector with +1 where we have information)
    """
    vector = np.zeros(vector_len)
    weight_vector = np.zeros(vector_len)
    for key, value in data.items():
        index = to_index_mapping[key]
        if value == 1:
            vector[index] = 1
            weight_vector = np.ones(vector_len)  # assume full information
        elif value == 0:  # TODO: fix this hack for now
            weight_vector[index] = 1
        else:
            raise ValueError("dict should be sparse, with just 1 and 0")
    return vector, weight_vector

# Just in the case of NOT multi-label (deprecated)
def get_split_dictionary(data):
    splits = []
    if len(data["incidents"]) == 0:
        for key, value in data["places"].items():
            splits.append({
                "incidents": {},
                "places": {key: value}
            })
    elif len(data["places"]) == 0:
        for key, value in data["incidents"].items():
            splits.append({
                "incidents": {key: value},
                "places": {}
            })
    else:
        for d, dv in data["incidents"].items():
            for p, pv in data["places"].items():
                splits.append({
                    "incidents": {d: dv},
                    "places": {p: pv}
                })
    return splits


# Class TestDataset build for test of Cifar100 and Oxford pet III
class TestDataset(Dataset):
    """A Pytorch dataset to simplify the Test implementation of default dataset

    Args:
        transform: Transformer used to process the images (from PIL to Tensor format)
        
    Attributes:
        all_data: List used to retrieve the single element (data plus all the vectors)
    """

    def __init__(self, transform=None):

        self.all_data = []
        print("Adding test images (only incidents F1-score additional performances)")

        # Parameter for the images folder
        folder_path = "./data"
        test_data = CIFAR100(root=folder_path, train=False, download=True)

        print(f"Zipped folder saved in {folder_path}")

        # This class is supposed to work with "sigmoid" layer (set all to 0)
        place_vector = np.zeros(49)
        incident_vector = np.zeros(43)
        place_weight_vector = np.zeros(49)
        incident_weight_vector = np.zeros(43)

        # PIL images in format: (PIL Image, label)
        for image in test_data:

          item = (image[0], place_vector, incident_vector,
                        place_weight_vector, incident_weight_vector)
          self.all_data.append(item)

        print("number items: {}".format(len(self.all_data)))
        # If there is need of a conversion (image loader)
        #self.image_loader = image_loader
        self.transform = transform

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: incident_label_item (list), no_incident_label_item (list)
        """
        my_item = list(self.all_data[index])
        img = my_item[0]
        if self.transform is not None:
            img = self.transform(img)
        my_item[0] = img
        return my_item

# _v2
class IncidentDataset_v2(Dataset):

  """A Pytorch dataset for multi_labels classification: images with incidents and places labels.

    Args:
        images_path (str): Path of the images folder
        incidents_images (dict): Images that are part of our dataset.
        place_to_index_mapping (dict): Dict of 'key:idx' pairs for the places
        incident_to_index_mapping (dict): Dict of 'key:idx' pairs for the incidents
        transform (list of Transform obj): Processing to apply to the images
        threads (int): Number of process to parallelize
        keep_track (int): Number of completed images considered for the status print during the parsing 

    Attributes:
        place_names (list): List of the place names.
        place_name_to_idx (dict): Dict with items (place_name, index).
        incident_name (list): List of the incident names.
        incident_name_to_idx (dict): Dict with items (incident_name, index).
    """ 
  def __init__(self,
              images_path,
              incidents_images,
              place_to_index_mapping,
              incident_to_index_mapping,
              transform=None,
              pos_only=False,
              threads = 10,
              keep_track=10):

              # Set internal variables (IMPORTANT: Tranformer and Image loader)
              self.images_path = images_path
              self.incidents_images = incidents_images # Dict of the images
              self.transform = transform
              self.image_loader = image_loader_v2

              logging.info(f"Starting loading images from path '{self.images_path}'")

              self.data = [] # Simple built-in list

              # Start parallelization of the actual data parsing from json file
              with ThreadPoolExecutor(max_workers = threads) as executor:
                futures = [executor.submit(self.get_parsed_data, path, incident_to_index_mapping, place_to_index_mapping, values["incidents"], values["places"]) for path, values in self.incidents_images.items()]
                
                job_completed = 0 # temporary solution to keep count of the jobs
                for future in as_completed(futures):
                  job_completed += 1
                  # Load the mapped/loaded image values in final list
                  item = future.result()
                  # For modularity It is added the 'pos_only' option for the training phase
                  if pos_only: # save the item if it contains at least one postive label for incident
                    if sum(item[2]) > 0:
                      self.data.append(item)
                  else:
                    self.data.append(item) # Standard list of Dict

                  # Keep track of the progress
                  if job_completed % keep_track == 0:
                    logging.debug(f"> Images already parsed: {job_completed}")

              # For the label 'no_incidents'/'no_places' It will be considered if the threshold is not respected -> validation phase of the classes scores
              logging.info(f"The number of items succesfully loaded are {len(self.data)}")
              

  # Modular function to parallelize
  @staticmethod
  def get_parsed_data(file_path, mapping_incidents, mapping_places, incidents, places):
    """For every image_path return the data item 

    Args: 
        mapping_incidents (dict): Every key (class name) has a corresponding int value
        mapping_places (dict): Every key (class name) has a corresponding int value
        incidents (dict): {"Incident": +1/0}
        places (dict): {"Place": +1/0}

    Returns:
        Data item: Built-in Dict that contain all the parsed values
    """
    # TO FIX: Return Tensors NOT ndarray --> for the torch.stack in data collator
    incident_vector, incident_weight_vector = get_vectors(incidents, mapping_incidents, len(mapping_incidents))
    place_vector, place_weight_vector = get_vectors(places, mapping_places, len(mapping_places))
     
    # Parsed item of the images: return a Dict as standard in the HuggingFace documentation
    return {'image':file_path, 'incidents_target':incident_vector, 'places_target':place_vector, 'incidents_weight':incident_weight_vector, 'places_weight':place_weight_vector}

  # Override of the class for the custom dataset (_v2)
  def __len__(self):
    return len(self.data)

  # Override of the class for the custom dataset (_v2)
  def __getitem__(self, index):
    """
    Args:

        index (int): Index to fetch from the list of items

        Returns:
            tuple: incident_label_item (list), no_incident_label_item (list)
    """
    my_item = self.data[index]
    image_name = my_item["image"]
    img = self.image_loader(os.path.join(self.images_path, image_name))
    if self.transform is not None:
        img = self.transform(img)
    my_item["image"] = img # Subscribe the path with the "processed" image
    return my_item

# _v2
def  get_datasets_v2(args):
  """Main function to return a Dataset with the customized inner class (IncidentDataset)

    Args:
        args (dict): Parsed arguments
        is_train (bool): True if the Dataloader will be used in training

    Return: Dataloader with the settings specified in the "mode" arg
    """
  is_train = True
  if args.mode == "test":
    is_train = False

  if args.download_train_json == "True":
    # If you require preprocessing, the args.dataset_train will be updated to the clean one
    args.dataset_train = download_images_from_json_parallelized(file_path=args.dataset_train, folder_path=args.images_path)

  if args.download_val_json == "True":
    download_images_from_json_parallelized(file_path=args.dataset_val, folder_path=args.images_path)

  # Mapping imported from utils.py using old function
  place_to_index_mapping = get_place_to_index_mapping()
  incident_to_index_mapping = get_incident_to_index_mapping()

  # Instantiate normalization for the transform composition (both train/val/test)
  normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
  # Set up the val/test transformer
  val_transform = transforms.Compose([
              transforms.Resize(416),
              transforms.CenterCrop(384),
              transforms.ToTensor(),
              normalize
              ])

  # Load the correct paths and pass to the function to create the custom dataset
  if is_train == True:
    
    # Set up the train transformer
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(384),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])

    logging.debug(f"Loading images path/values from file for train/validation splits ({args.dataset_train})")
    # Retrived the json data (images_path: values)
    train_val_dict = get_loaded_json_file_v2(args.dataset_train)

    # Use the train file to create train/val split
    train_keys, val_keys = get_train_idx(train_val_dict)
    train_dict = dict((k, train_val_dict[k]) for k in train_keys)
    val_dict = dict((k, train_val_dict[k]) for k in val_keys)
    logging.info(f"Images name/values correctly loaded: train {len(train_dict.keys())} samples   val {len(val_dict.keys())} samples")
    logging.info(f"Currently working on train dataset ..")

    if args.examples == "pos_and_neg": # Checking the image examples to use
      pos_only = False
    else:
      pos_only = True

    train_dataset = IncidentDataset_v2(args.images_path, train_dict, place_to_index_mapping, incident_to_index_mapping, train_transform, pos_only)
    val_dataset = IncidentDataset_v2(args.images_path, val_dict, place_to_index_mapping, incident_to_index_mapping, val_transform, pos_only) # For now val_dataset is instantiated as train_dataset even if the weight vesctors are not needed

    return train_dataset, val_dataset # In case of training phase

  return  None, test_dataset # In case of test phase

class IncidentDataset(Dataset):
    """A Pytorch dataset for classification of incidents images with incident and place.

    Args:
        incidents_images (dict): Images that are part of our dataset.
        place_to_index_mapping (dict):
        incident_to_index_mapping (dict):

    Attributes:
        place_names (list): List of the place names.
        place_name_to_idx (dict): Dict with items (place_name, index).
        incident_name (list): List of the incident names.
        incident_name_to_idx (dict): Dict with items (incident_name, index).
    """

    def __init__(
            self,
            images_path,
            incidents_images,
            place_to_index_mapping,
            incident_to_index_mapping,
            transform=None,
            use_all=False,
            pos_only=False,
            using_softmax=False,
            use_multi_label=True):


        self.images_path = images_path
        self.use_all = use_all

        self.items = []
        self.all_data = []
        self.no_incident_label_items = []  # items without a incident label

        self.no_incidents = defaultdict(list)

        print("adding incident images")
        for filename, original_data in tqdm(incidents_images.items()):

            if not use_multi_label:
                splits = get_split_dictionary(original_data)
            else:
                splits = [copy.deepcopy(original_data)]
            for data in splits:
                
                if not use_multi_label:
                    assert len(data["incidents"]) <= 1 and len(data["places"]) <= 1

                if using_softmax:
                    # the +1 to len accounts for "no place" and "no incident"
                    place_vector, place_weight_vector = get_vectors(
                        data["places"], place_to_index_mapping, len(place_to_index_mapping) + 1)
                    incident_vector, incident_weight_vector = get_vectors(
                        data["incidents"], incident_to_index_mapping, len(incident_to_index_mapping) + 1)
                else:
                    place_vector, place_weight_vector = get_vectors(
                        data["places"], place_to_index_mapping, len(place_to_index_mapping))
                    incident_vector, incident_weight_vector = get_vectors(
                        data["incidents"], incident_to_index_mapping, len(incident_to_index_mapping))

                # TODO: need to add "no incident" to some...
                # TODO: somehow fix this hack
                # means its part of the places dataset, so no incident
                if len(data["incidents"]) == 0 and using_softmax == True:
                    incident_vector[-1] = 1  # "no incident" is +1
                    incident_weight_vector = np.ones(
                        len(incident_weight_vector))
                elif len(data["incidents"]) == 0:
                    incident_vector = np.zeros(len(incident_weight_vector))
                    incident_weight_vector = np.ones(
                        len(incident_weight_vector))

                # choose which set to put them into
                has_incident = False
                for label in data["incidents"].values():
                    if label == 1:
                        has_incident = True
                        break

                item = (filename, place_vector, incident_vector,
                        place_weight_vector, incident_weight_vector)

                if pos_only:
                    if has_incident:
                        self.all_data.append(item)
                    else:
                        pass
                else:
                    self.all_data.append(item)

        print("number items: {}".format(len(self.all_data)))
        self.image_loader = image_loader
        self.transform = transform

    def __len__(self):
        return len(self.all_data)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: incident_label_item (list), no_incident_label_item (list)
        """

        my_item = list(self.all_data[index])
        image_name = my_item[0]
        img = image_loader(os.path.join(self.images_path, image_name))
        if self.transform is not None:
            img = self.transform(img)
        my_item[0] = img
        return my_item


# TODO: change to dataloader, not dataset
def get_dataset(args,
                is_train=True,
                is_test=False):
    """

    :param args:
    :param is_train:
    :param is_test:
    :return:
    """
    # """Returns the dataset for training or testing.
    #
    # Args:
    #     args:
    #
    # Returns:
    #     DataLoader:
    # """

    # main dataset (incidents images)
    if is_train:
        incidents_images = get_loaded_json_file(args.dataset_train)
        idx = int(len(incidents_images) * args.percent_of_training_set / 100.0)
        keys = list(incidents_images.keys())[:idx]
        print(len(keys))
        incidents_images_temp = {}
        for key in keys:
            incidents_images_temp[key] = incidents_images[key]
        incidents_images = incidents_images_temp
    else:
        if is_test == False:  # validation images

            # Addition for testing cifar100 and Oxford pet
            if args.dataset_val == "cifar100":
              # Duplicate of variables
              normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
              transform = transforms.Compose([
              transforms.Resize(416),
              transforms.CenterCrop(384),
              transforms.ToTensor(),
              normalize])
              # Init custom dataset
              dataset = TestDataset(transform)
              # Init and return the standard loader
              loader = torch.utils.data.DataLoader(
              dataset,
              batch_size=args.batch_size,
              shuffle=False,
              num_workers=args.workers,
              pin_memory=True
              )
              return loader
              
            incidents_images = get_loaded_json_file(args.dataset_val)
        else:  # test images
            incidents_images = get_loaded_json_file(args.dataset_test)

    place_to_index_mapping = get_place_to_index_mapping()
    incident_to_index_mapping = get_incident_to_index_mapping()

    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    if is_train:
        pos_only = args.dataset == "pos_only"
        shuffle = True
        use_all = False
        transform = transforms.Compose([
            transforms.RandomResizedCrop(384),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize
        ])
    else:
        pos_only = False
        shuffle = False
        use_all = True
        transform = transforms.Compose([
            transforms.Resize(416),
            transforms.CenterCrop(384),
            transforms.ToTensor(),
            normalize,
        ])

    using_softmax = args.activation == "softmax"

    dataset = IncidentDataset(
        args.images_path,
        incidents_images,
        place_to_index_mapping,
        incident_to_index_mapping,
        transform=transform,
        use_all=use_all,
        pos_only=pos_only,
        using_softmax=using_softmax
    )

    # TODO: avoid the double shuffling affect that currently exists
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True
    )

    return loader
