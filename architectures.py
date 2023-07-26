"""
models.py

- Set up the customized model architecture (backbone from HuggingFace)
- Set up the customized Trainer class from HuggingFace
"""
from PIL import Image
from torch.nn import functional as F
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data as data
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import os
import wget
import timm
import tensorflow as tf
from transformers import Trainer, ViTConfig, ViTModel
import logging
from utils import image_loader

# _v2 Compose used for trial of the backbone/ensemble 
transform = transforms.Compose([
            transforms.RandomResizedCrop(224), # Dimension of the transformer input required
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
        ])

# Old loader used during training (deprecated)
inference_loader = transforms.Compose([
    transforms.Resize((416, 416)),
    transforms.CenterCrop(384),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# _v2
class IncidentModel(nn.Module):
  """ Function to attach multiple classification heads to the original 'trunk' model from HuggingFace

      return: Ensemble model created with standard 'PyTorch' API
  """
  def __init__(self, backbone, in_features, incident_out_features, places_out_features):
    super(IncidentModel, self).__init__()
    self.backbone = backbone
    self.incident_head = nn.Linear(in_features, incident_out_features, bias=True, device=None, dtype=None)
    self.place_head = nn.Linear(in_features, places_out_features, bias=True, device=None, dtype=None)
        
  def forward(self, input_data):
    backbone_output = self.backbone(input_data) # Backbone from HuggingFace provide a final dense layer with the tanh activation function
    incident_output = self.incident_head(backbone_output[0])
    place_output = self.place_head(backbone_output[0])

    return incident_output, place_output # Sigmoid applied on the 'loss computation'

#_ v2 (TO FIX)
def save_model_architecture(args):
  """ Function to save architecture config (called from get model)
  
  """
  return

# _v2
def get_model(args, device):
  """ Function to get the backbone from HuggingFace library and add the two heads
  
  """
  if args.arch == "google_vit_b_16":
    if args.load_arch_path != "None":
      # Check if the custom architecture config file 'actually' exist
      if os.path.exists(args.load_arch_path):
        backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", config=args.load_arch_path)
    else: # Load the predefined optimal tranformer config
      logging.debug(f"Can't find the architecture path: {args.load_arch_path} - load the default model config.")
      backbone = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k", config = ViTConfig())
  
  logging.info(f"Backbone {args.arch} succefully loaded (from HuggingFace)")
  logging.debug(f"Backbone architecture: \n{backbone}") 

  # Test the backbone
  image = image_loader(args.trial_image)
  image_tensor = transform(image)
  image_tensor = image_tensor[None, :] # Added a dummy dim for the trial 'batch' of one image 
  with torch.no_grad():
    outputs = backbone(image_tensor)
  logging.info(f"Backbone is working: Output shape {outputs[0].size()}   -   Last hidden layer dim {outputs[0].size()[-1]}")
  last_hidden_layer_dim = outputs[0].size()[-1] # Get hidden layer dim dinamically
  
  # Create the ensemble
  model = IncidentModel(backbone, last_hidden_layer_dim, args.num_incidents, args.num_places)
  logging.info(f"Custom model succesfully loaded\n{model}")

  with torch.no_grad(): # Test the ensemble model
    outputs = model(image_tensor)
  
  logging.info(f"Model is working: Output shape {outputs[0].size()}   -   Last hidden layer dim {outputs[0].size()[-1]}")
  
  # Check for already saved weights (TO FIX)
  model.to(device)

  return model

# Old function for the 'demo_server.py'
class FilenameDataset(data.Dataset):
    """
    Data loader for filenames and their corresponding labels.
    """

    def __init__(self, image_filenames, targets):
        """
        Args:
            image_filenames (list): List of image filenames
            targets (list): List of integers that correspond to target class indices
        """
        assert (len(image_filenames) == len(targets))
        self.image_filenames = image_filenames
        self.targets = targets

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is the target class index
        """
        image_filename = self.image_filenames[index]
        # if not os.path.isfile(image_filename):
        #     os.system("ln -s {} {}".format(image_filename.replace("/data/vision/torralba/humanitarian/datasets/images_raw/",
        #                                                           "/data/vision/torralba/humanitarian/dimitris/getGoogleImages2/finalImages/"), image_filename))
        if not os.path.isfile(image_filename):
            raise ValueError("{} is not a file".format(image_filename))
        try:
            with open(image_filename, 'rb') as f:
                image = Image.open(f).convert('RGB')
                image = inference_loader(image)
        except:
            print(image_filename)
            image = Image.new('RGB', (300, 300), 'white')
            image = inference_loader(image)
        return image, self.targets[index]

    def __len__(self):
        return len(self.image_filenames)

# Modified by US for the ViT part
def get_trunk_model(args):
    
    #VIT-B-16 TRIAL UNFREEZED
    if args.arch == "vit_b_16":
      link_root = "https://storage.googleapis.com/vit_models/augreg/"
      filename = 'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz'
      model_name = 'vit_base_patch16_384'
      model = timm.create_model(model_name, num_classes=1024)

      # Non-default checkpoints need to be loaded from local files.
      if not tf.io.gfile.exists(filename):
        print('Pre-trained weights not found. Downloading...')
        print(link_root+filename)
        wget.download(link_root + filename)
      timm.models.load_checkpoint(model, filename)
      print("**** Loaded ViT pre-trained ****")
      
      for param in model.parameters():
          param.requires_grad = False
      for block in model.blocks:
            for param in block.attn.parameters():
                param.requires_grad = True
      for param in model.head.parameters():
            param.requires_grad = True
      print("**** Model loaded and freezed, except MHSA and last linear ****")
      return model


    #VIT-B-16
    if args.arch == "vit_b_16":
      link_root = "https://storage.googleapis.com/vit_models/augreg/"
      filename = 'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz'
      model_name = 'vit_base_patch16_384'
      model = timm.create_model(model_name, num_classes=1024)

      # Non-default checkpoints need to be loaded from local files.
      if not tf.io.gfile.exists(filename):
        print('Pre-trained weights not found. Downloading...')
        print(link_root+filename)
        wget.download(link_root + filename)
      timm.models.load_checkpoint(model, filename)
      print("**** Loaded ViT pre-trained ****")
      for name, child in model.named_children():
            #print(f"Name: {name}")
            #print(f"child: {child}")
            if name.startswith("head"):
                print("Trainable block: ", child)
                break
            for params in child.parameters():
                params.requires_grad = False
      print("**** Model loaded and freezed, except last layer ****")
      return model
    
    # VIT-L-16
    if args.arch == "vit_l_16":
      link_root = "https://storage.googleapis.com/vit_models/augreg/"
      filename = 'L_16-i21k-300ep-lr_0.001-aug_strong1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz'
      model_name = 'vit_large_patch16_384'
      model = timm.create_model(model_name, num_classes=1024)

      # Non-default checkpoints need to be loaded from local files.
      if not tf.io.gfile.exists(filename):
        print('Pre-trained weights not found. Downloading...')
        print(link_root+filename)
        wget.download(link_root + filename)
      timm.models.load_checkpoint(model, filename)
      print("**** Loaded ViT pre-trained ****")

      for name, child in model.named_children():
            #print(f"Name: {name}")
            #print(f"child: {child}")
            if name.startswith("head"):
                print("Trainable block: ", child)
                break
            for params in child.parameters():
                params.requires_grad = False
      print("**** Model loaded and freezed, except last layer ****")
      return model

    if args.pretrained_with_places:
        print("loading places weights for pretraining")
        model = models.__dict__[args.arch](num_classes=365)
        dir_path = os.path.dirname(os.path.realpath(__file__))
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.arch == "resnet18":
            model_file = os.path.join(dir_path, "pretrained_weights/resnet18_places365.pth.tar")
            checkpoint = torch.load(model_file, map_location=device)
            state_dict = {str.replace(k, 'module.', ''): v for k,
                                                               v in checkpoint['state_dict'].items()}
            model.load_state_dict(state_dict)
            model.fc = nn.Linear(512, 1024)
            model = nn.Sequential(model, nn.ReLU())
        elif args.arch == "resnet50":
            model_file = os.path.join(dir_path, "pretrained_weights/resnet50_places365.pth.tar")
            checkpoint = torch.load(model_file, map_location=device)
            state_dict = {str.replace(k, 'module.', ''): v for k,
                                                               v in checkpoint['state_dict'].items()}
            model.load_state_dict(state_dict)
            model.fc = nn.Linear(2048, 1024)
            model = nn.Sequential(model, nn.ReLU())
        return model
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dir_path = os.path.dirname(os.path.realpath(__file__))
        print("loading imagenet weights for pretraining")
        # otherwise load with imagenet weights
        if args.arch == "resnet18":
            model = models.resnet18(pretrained=True)
            for name, child in model.named_children():
              #print(f"Name: {name}")
              #print(f"child: {child}")
              if name=="fc":
                pass
                print("Trainable block: ", child)
              for params in child.parameters():
                  params.requires_grad = False
            model.fc = nn.Sequential(nn.Linear(512, 1024), nn.ReLU())

        elif args.arch == "resnet50":
            print("*** Using ResNet50 with Imagenet21k pre-trained weights ***")
            
	    # START MODIFICATION
            model = models.resnet50(weights=None)
            model_file = os.path.join(dir_path, "pretrained_weights/resnet50_pretrain_im21k.pth")
            original_dict = torch.load(model_file, map_location=device)["state_dict"]
            state_dict = {}
            # Compose the state_dict (mismatch between layers)
            for key in original_dict.keys():
              if key.split(".")[-1] != 'num_batches_tracked':
                state_dict[key] = original_dict[key]

            # DEBUG 
            #filename = '/content/IncidentsDataset/pretrained_weights/resnet50_places365.pth.tar'
            #checkpoint_example = torch.load(filename, map_location=device)
            #state_dict_tmp = {str.replace(k, 'module.', ''): v for k, v in checkpoint_example['state_dict'].items()}
            #print(state_dict_tmp.keys())

            # Load random tensor and vector to compose the last layer of the resnet (fc was modified from the repository, only the final number of classes)
            state_dict["fc.weight"]= torch.rand(size=(1000, 2048))
            state_dict["fc.bias"]= torch.rand(size=(1000,))
            model.load_state_dict(state_dict)
            print("Model loaded correctly!")
            for name, child in model.named_children():
              if name=="fc":
                print(f"Trainable block: {child}")
                break
              for params in child.parameters():
                  params.requires_grad = False
            #model.fc = nn.Linear(2048, 1024)
            #model = nn.Sequential(model, nn.ReLU())
            model.fc = nn.Sequential(nn.Linear(2048, 1024), nn.ReLU())
        return model

def get_incident_layer(args):
    if args.activation == "softmax":
        return nn.Linear(args.fc_dim, args.num_incidents + 1)
    elif args.activation == "sigmoid":
        return nn.Linear(args.fc_dim, args.num_incidents)


def get_place_layer(args):
    if args.activation == "softmax":
        return nn.Linear(args.fc_dim, args.num_places + 1)
    elif args.activation == "sigmoid":
        return nn.Linear(args.fc_dim, args.num_places)


def get_incidents_model(args):
    """
    Returns [trunk_model, incident_layer, place_layer]
    """
    # the shared feature trunk model
    trunk_model = get_trunk_model(args)
    # the incident model
    incident_layer = get_incident_layer(args)
    # the place model
    place_layer = get_place_layer(args)

    print("Let's use", args.num_gpus, "GPUs!")
    trunk_model = torch.nn.DataParallel(trunk_model, device_ids=range(args.num_gpus))
    incident_layer = torch.nn.DataParallel(incident_layer, device_ids=range(args.num_gpus))
    place_layer = torch.nn.DataParallel(place_layer, device_ids=range(args.num_gpus))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    trunk_model.to(device)
    incident_layer.to(device)
    place_layer.to(device)
    return [trunk_model, incident_layer, place_layer]


def update_incidents_model_with_checkpoint(incidents_model, args):
    """
    Update incidents model with checkpoints (in args.checkpoint_path)
    """

    trunk_model, incident_layer, place_layer = incidents_model

    # optionally resume from a checkpoint
    # TODO: bring in the original pretrained weights maybe?
    # TODO: remove the args.trunk_resume, etc.
    # TODO: remove path prefix

    config_name = os.path.basename(args.config)
    
    best_str = "_best" if args.mode == "test" else ""

    trunk_resume = os.path.join(
        args.checkpoint_path, "trunk{}.pth.tar".format(best_str))
    place_resume = os.path.join(
        args.checkpoint_path, "place{}.pth.tar".format(best_str))
    incident_resume = os.path.join(
        args.checkpoint_path, "incident{}.pth.tar".format(best_str))

    # trunk_resume = "/data/vision/torralba/scratch/ethanweber/DamageAssessment/external/IncidentsDataset/pretrained_weights/eccv_final_model_trunk.pth.tar"
    # place_resume = "/data/vision/torralba/scratch/ethanweber/DamageAssessment/external/IncidentsDataset/pretrained_weights/eccv_final_model_place.pth.tar"
    # incident_resume = "/data/vision/torralba/scratch/ethanweber/DamageAssessment/external/IncidentsDataset/pretrained_weights/eccv_final_model_incident.pth.tar"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for (path, net) in [(trunk_resume, trunk_model), (place_resume, place_layer), (incident_resume, incident_layer)]:
        if os.path.isfile(path):
            checkpoint = torch.load(path, map_location=device)
            args.start_epoch = checkpoint['epoch']
            net.load_state_dict(checkpoint['state_dict'])
            print("Loaded checkpoint '{}' (epoch {}).".format(path, checkpoint['epoch']))
        else:
            print("No checkpoint found at '{}'.".format(path))


def update_incidents_model_to_eval_mode(incidents_model):
    print("Switching to eval mode.")
    for m in incidents_model:
        # switch to evaluation mode
        m.eval()


def get_predictions_from_model(args,
                               incidents_model,
                               batch_input,
                               image_paths,
                               index_to_incident_mapping,
                               index_to_place_mapping,
                               inference_dict, topk=1):
    """
    Input:
    {
        "image_paths" = [list of image paths],
    }
    Returns {
        "incidents": [], # list of topk elements
        "places": [] # list of topk elements
    }
    """
    trunk_model, incident_layer, place_layer = incidents_model

    # compute output with models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = batch_input.to(device)
    output = trunk_model(input)
    incident_output = incident_layer(output)
    place_output = place_layer(output)

    if args.activation == "softmax":
        incident_output = F.softmax(incident_output, dim=1)
        place_output = F.softmax(place_output, dim=1)
    elif args.activation == "sigmoid":
        m = nn.Sigmoid()
        incident_output = m(incident_output)
        place_output = m(place_output)

    incident_probs, incident_idx = incident_output.sort(1, True)
    place_probs, place_idx = place_output.sort(1, True)

    temp_inference_dict = {}

    # batch_input[0] is the batch dimension (the # in the batch)
    for batch_idx in range(len(batch_input.numpy())):
        incidents = []
        for idx in incident_idx[batch_idx].cpu().numpy()[:topk]:
            if idx < len(index_to_incident_mapping):
                incidents.append(
                    index_to_incident_mapping[idx]
                )
            else:
                incidents.append("no incident")

        places = []
        for idx in place_idx[batch_idx].cpu().numpy()[:topk]:
            if idx < len(index_to_place_mapping):
                places.append(
                    index_to_place_mapping[idx]
                )
            else:
                places.append("no place")

        output = {
            "incidents": incidents,
            "places": places,
            "incident_probs": incident_probs[batch_idx].cpu().detach().numpy()[:topk],
            "place_probs": place_probs[batch_idx].cpu().detach().numpy()[:topk]
        }
        image_path = image_paths[batch_idx]
        temp_inference_dict[image_path] = output

    # TODO: maybe return the output here
    if inference_dict is not None:
        inference_dict.update(temp_inference_dict)
    return temp_inference_dict


def get_predictions_from_model_all(args, incidents_model, batch_input, image_paths, index_to_incident_mapping,
                                   index_to_place_mapping, inference_dict, softmax=True):
    """
    Input:
    {
        "image_paths" = [list of image paths],
    }
    Returns {
        "incidents": [], # list of topk elements
        "places": [] # list of topk elements
    }
    """
    trunk_model, incident_layer, place_layer = incidents_model

    # compute output with models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = batch_input.to(device)
    output = trunk_model(input)
    incident_output = incident_layer(output)
    place_output = place_layer(output)

    if softmax:
        incident_output = F.softmax(incident_output, dim=1)
        place_output = F.softmax(place_output, dim=1)
    else:
        m = nn.Sigmoid()
        incident_output = m(incident_output)
        place_output = m(place_output)

    incident_probs, incident_idx = incident_output.sort(1, True)
    place_probs, place_idx = place_output.sort(1, True)

    # batch_input[0] is the batch dimension (the # in the batch)
    for batch_idx in range(len(batch_input.numpy())):
        incidents = []
        for idx in incident_idx[batch_idx].cpu().numpy():
            if idx < len(index_to_incident_mapping):
                incidents.append(
                    index_to_incident_mapping[idx]
                )
            else:
                incidents.append("no incident")

        places = []
        for idx in place_idx[batch_idx].cpu().numpy():
            if idx < len(index_to_place_mapping):
                places.append(
                    index_to_place_mapping[idx]
                )
            else:
                places.append("no place")

        output = {
            "incidents": incidents,
            "places": places,
            "incident_probs": incident_probs[batch_idx].cpu().detach().numpy(),
            "place_probs": place_probs[batch_idx].cpu().detach().numpy()
        }
        image_path = image_paths[batch_idx]
        inference_dict[image_path] = output

    # TODO: maybe return the output here
    return None


def get_features_from_model(incidents_model, batch_input, image_paths, inference_dict):
    """
    Input:
    {
        "image_paths" = [list of image paths],
    }
    Returns trunk_model output.
    """
    trunk_model, incident_layer, place_layer = incidents_model

    # compute output with models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input = batch_input.to(device)
    output = trunk_model(input)

    # batch_input[0] is the batch dimension (the # in the batch)
    for batch_idx in range(len(batch_input.numpy())):
        out = output[batch_idx].cpu().detach().numpy()
        # print("here")
        # print(out)
        # print(out.shape)
        # print(type(out))
        image_path = image_paths[batch_idx]
        inference_dict[image_path] = out

    # TODO: maybe return the output here
    return None
