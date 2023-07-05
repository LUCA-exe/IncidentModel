import os
from PIL import Image
from tqdm import tqdm
import json

from multiprocessing.dummy import Pool as ThreadPool

images_path = "data/images/"

def inspect_image(image):    
    try:
        im = Image.open(images_path + image)
        im.verify()
        im.close()
    except Exception:
        return False

    return True

def main():

    # Get all images in the images folder
    images = [img for img in os.listdir(images_path) if img.endswith(".jpg")]

    pool = ThreadPool(20)
    results = list(tqdm(pool.imap(inspect_image, images), total=len(images)))
    pool.close()
    pool.join()

    # Map the images to results
    results = dict(zip(images, results))
    print(results)

    # Save results to json
    with open("data/image_status.json", "w") as f:
        json.dump(results, f, indent=4)

if __name__ == "__main__":
    main()

