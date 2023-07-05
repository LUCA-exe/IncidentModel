import requests
from PIL import Image
import json
import os
import configargparse

# Custom parser
def get_parser():
    parser = configargparse.ArgumentParser(description="Incident Model scraper program.")
    parser.add_argument("--path_folder",
                        default="IncidentsDataset/data",
                        type=str,
                        help="Insert the absolut path of the folder containing the json file containing the images info.")
    parser.add_argument("--name_file",
                        default="eccv_val.json",
                        type=str,
                        help="Insert the name of the json file containing the images info.")
    parser.add_argument("--images_folder",
                        default="images_eccv_val",
                        type=str,
                        help="The images will be created in the 'path folder'.")
    parser.add_argument("--n_images",
                        default=10,
                        type=int,
                        help="How many images you want to check.")                    
    return parser

def download_images(args):
  string_path = os.path.join(args.path_folder, args.name_file)
  print(string_path)
  # Clean json saving path
  new_string_path = os.path.join(args.path_folder, "new_"+args.name_file)

  with open(string_path, "r") as fp:
    dataset = json.load(fp)
  new_dataset = {}
  cont = 0
  cont_error = 0
  for image_name in dataset.keys():
    cont += 1
    img_url = dataset[image_name]["url"]
    try:
        img = Image.open(requests.get(img_url, stream=True).raw)
        img.save(os.path.join(args.path_folder, args.images_folder, image_name.split("/")[0] + "_" + image_name.split("/")[1]))
        new_dataset[image_name.split("/")[0] + "_" + image_name.split("/")[1]] = dataset[image_name]
    except Exception as e:
        cont_error += 1
        print(str(e))
    if cont == args.n_images:
        break
  print(f"Debug: Broken link found while loading images: {cont_error} - {round((cont_error / args.n_images) * 100, 2)}%")
  # Save the clean json
  try:
    with open(new_string_path, 'w') as fp:
      json.dump(new_dataset, fp)
      print(f">>> Saved {new_string_path} correctly <<<")
  except Exception as e:
      print(str(e))
  return


if __name__ == '__main__':
  args = get_parser().parse_args()
  download_images(args)

# Creare cicli diversi per ---> nre_json misto, new_json only_positive, new_json only negative
















