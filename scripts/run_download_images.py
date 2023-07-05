import pandas as pd
import requests
import uuid

from tqdm import tqdm
from multiprocessing.dummy import Pool as ThreadPool

def get_image_url(row):
    return row["url"]


def download_image_from_url(url):
    uuid_str = str(uuid.uuid4())
    try:
        r = requests.get(url, allow_redirects=True, timeout=3)

        if r.status_code == 200:
            # download the image
            open("data/images/" + uuid_str + ".jpg", "wb").write(r.content)
            return 200
        else:
            return r.status_code
    except Exception:
        return -1

def download_images(dataset_name, limit=-1):
    path = f"data/{dataset_name}.pkl"
    print(f"Loading {dataset_name} data from {path}...")

    df = pd.read_pickle(path)

    if limit > 0:
        df = df.sample(limit, random_state=limit)

    print("Running image url download...")
    urls = df["url"]

    # Download images using multithreading with progress report
    pool = ThreadPool(40)
    results = list(tqdm(pool.imap(download_image_from_url, urls), total=len(urls)))
    pool.close()
    pool.join()

    df["status_code"] = results

    # Save df to pickle
    df.to_pickle(f"{dataset_name}_status_codes.pkl")

    # Save df to json with line formatting
    df.to_json(f"{dataset_name}_status_codes.json", orient="records", lines=True)



def main():
    
    download_images("multi_label_train")
    download_images("multi_label_val")

if __name__ == "__main__":
    main()