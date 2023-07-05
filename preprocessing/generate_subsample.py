import pandas as pd
import json
import argparse
import os
import shutil

from tqdm import tqdm
from typing import List


def save_df(df: pd.DataFrame, output_file: str):
    # Save df to json with line formatting
    values_dict = df[["incidents", "places"]].to_dict(orient="records")
    final_dict = {k: v for k, v in zip(df["key"], values_dict)}

    # Save final_dict to json
    with open(output_file, "w") as f:
        json.dump(final_dict, f)

    print(f"Saved {len(final_dict)} samples to {output_file} successfully.")


def filter_df(df: pd.DataFrame, ignore_incidents: List[str], ignore_places: List[str], num_samples: int = -1):
    df = df[~df["incidents_list"].apply(lambda x: any([i in ignore_incidents for i in x.split(", ")]))].copy()
    df = df[~df["places_list"].apply(lambda x: any([i in ignore_places for i in x.split(", ")]))].copy()

    if num_samples > 0:
        df = df.sample(n=num_samples, random_state=42).copy()
    return df


def copy_images_from_df(df: pd.DataFrame, output_dir: str):
    base_image_dir = os.path.join(os.path.join(os.path.sep.join(output_dir.split(os.path.sep)[:-1])), "images")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Copying images from {base_image_dir} to {output_dir}...")
    for image_id in tqdm(df["key"]):
        shutil.copy(os.path.join(base_image_dir, image_id), os.path.join(output_dir, image_id))


def main():
    # Read in arguments from Argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle-file', '-p', type=str, required=True, help='Path to input pickle file')
    parser.add_argument('--output-file', '-o', type=str, default="./data/train.json",
                        help='Path to output json file')
    parser.add_argument('--num-samples', type=int, default=-1, help='Number of samples to be included in the dataset')
    parser.add_argument("--copy-images", "-c", action="store_true", default=False,
                        help="Copy images to a subsample output directory located in the same directory as the images directory")
    parser.add_argument("--train-validation-split", '-s', type=float, default=1.0,
                        help="Percentage of data to be used for training")
    parser.add_argument("--ignore-incidents", '-ii', type=str, nargs="+",
                        default=[], help="List of incidents to be ignored")
    parser.add_argument("--ignore-places", '-ip', type=str, nargs="+", default=[], help="List of places to be ignored")
    args = parser.parse_args()

    print(f"Ignoring incidents: {args.ignore_incidents}")
    print(f"Ignoring places: {args.ignore_places}")

    df = pd.read_pickle(args.pickle_file)
    df = df[df["valid_image"] == True].copy()

    if args.copy_images and args.num_samples > 0:
        copy_images_path = os.path.join(os.path.join(os.path.sep.join(
            args.output_file.split(os.path.sep)[:-1])), f"subsample_images")

    df = filter_df(df, args.ignore_incidents, args.ignore_places, args.num_samples)
    print(f"Filtered df to {len(df)} samples.")

    if args.train_validation_split < 1.0:
        print("Train-validation split is set to {}.".format(args.train_validation_split))
        print("The validation set will be saved in a separate file.")

        val_df = df.sample(frac=1 - args.train_validation_split, random_state=42)
        save_df(val_df, args.output_file.replace("train.json", "val.json"))
        if args.copy_images and args.num_samples > 0:
            copy_images_from_df(val_df, copy_images_path)
        df = df.drop(val_df.index)

    save_df(df, args.output_file)
    if args.copy_images and args.num_samples > 0:
        copy_images_from_df(df, copy_images_path)


if __name__ == "__main__":
    main()
