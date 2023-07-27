"""
Custom Trainer class to implement the custom loss for the Incident model

- Override the loss function 
"""
from torch import nn
from transformers import Trainer
import torch
import torch.nn as nn

# _v2
def data_collator(batch):
  """ Custom data_collator to form a batch of data for the modified compute_loss
  
  """
  return {'pixel_values': torch.stack([x['image'] for x in batch]),
          'incidents_targets': torch.stack([x['incidents_target'] for x in batch]),
          'places_targets': torch.stack([x['places_target'] for x in batch]),
          'incidents_weights': torch.stack([x['incidents_weight'] for x in batch]),
          'places_weights': torch.stack([x['places_weight'] for x in batch])
        }

# _v2 
class CustomTrainer(Trainer): # NOTE: Check for arguments on the Trainer args instead of override the classes for custom behaviour!
  """ Custom trainer (subclassed from Trainer original class)
  
  """
  def __init__(self, device, model, args, data_collator, train_dataset, eval_dataset): # Customize the Trainer with more internal variables
    super(CustomTrainer, self).__init__(model, args, data_collator, train_dataset, eval_dataset)
    self.device = device

  # Override
  def compute_loss(self, model, inputs, return_outputs=False):
    """ Compute loss: get the batch of data as Dict from the data collator
  
    """
    # Parse the data provided by the data collator Dict (in batch)
    incidents_targets, places_targets = inputs["incidents_targets"], inputs["places_targets"]
    incidents_weights, places_weights = inputs["incidents_weights"], inputs["places_weights"]
    incidents_outputs, places_outputs = model(inputs["pixel_values"])

    activation = nn.Sigmoid() # Instantiate the sigmoid layer

    incidents_outputs = activation(incidents_outputs) # [B, 43]
    places_outputs = activation(places_outputs) # [B, 49]

    # Check the formula from the original research
    criterion = nn.BCELoss(reduction='none')

    if self.device == "cpu":

      incident_loss = torch.sum(
        criterion(
          incidents_outputs,
          incidents_targets.type(torch.FloatTensor)
        ) * incidents_weights, dim=1) # to shape [B] 

      place_loss = torch.sum(
        criterion(
          places_outputs,
          places_targets.type(torch.FloatTensor)
        ) * places_weights, dim=1)

    else:

      incident_loss = torch.sum(
        criterion(
          incidents_outputs,
          incidents_targets.type(torch.FloatTensor).cuda(non_blocking=True)
        ) * incidents_weights, dim=1) # to shape [B] 

      place_loss = torch.sum(
        criterion(
          places_outputs,
          places_targets.type(torch.FloatTensor).cuda(non_blocking=True)
        ) * places_weights, dim=1)

    place_loss = place_loss.mean()
    incident_loss = incident_loss.mean()

    loss = incident_loss + place_loss
    # Following the standard format from HuggingFace library
    return (loss, (incidents_outputs, places_outputs)) if return_outputs else loss


  
  # Override: https://github.com/huggingface/transformers/blob/v4.31.0/src/transformers/trainer.py#L3260
  def prediction_step(self, model, inputs, prediction_loss_only: bool):

    with torch.no_grad():
      loss = None
      with self.compute_loss_context_manager():
        outputs = model(**inputs)
                    
    if prediction_loss_only:
      return (loss, None, None)


    return (loss, logits, labels)



def get_loss(args,
             incident_output,
             incident_target,
             incident_weight,
             place_output,
             place_target,
             place_weight):
    """
    
    Args:
        args: 
        incident_output: tensor of logits 
        incident_target: tensor with 1s and 0s representing the GT label (default 0)
        incident_weight: tensor with 1s where we have information
        place_output: 
        place_target: 
        place_weight: 
        is_train: 

    Returns:
        torch.Tensor: a scalar for the loss
    """

    # pass through desired activation
    if args.activation == "softmax":
        m = nn.Softmax(dim=1)
    elif args.activation == "sigmoid":
        m = nn.Sigmoid()
    incident_output = m(incident_output) # [B, 43]
    place_output = m(place_output) # [B, 49]

    criterion = nn.BCELoss(reduction='none')
    incident_loss = torch.sum(
        criterion(
            incident_output,
            incident_target.type(torch.FloatTensor).cuda(non_blocking=True)
        ) * incident_weight, dim=1) # to shape [B]
    # amplify the loss by the number of positive labels
    # if no positive labels, then multiply by ones
    # multiplier = torch.clamp(torch.sum(incident_target, dim=1), min=1)
    # incident_loss = (incident_loss * multiplier).mean()
    incident_loss = incident_loss.mean()

    place_loss = torch.sum(
        criterion(
            place_output,
            place_target.type(torch.FloatTensor).cuda(non_blocking=True)
        ) * place_weight, dim=1)
    # multiplier = torch.clamp(torch.sum(place_target, dim=1), min=1)
    # place_loss = (place_loss * multiplier).mean()
    place_loss = place_loss.mean()
    loss = incident_loss + place_loss
    return loss, incident_output, place_output
