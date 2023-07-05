import pandas as pd
import json
import os

def delete_images(df, path_to_images):
    delete_images_df = df[(df["valid_image"] == False) & (df["downloadable"] == True)]
    deleted_images = 0
    for index, row in delete_images_df.iterrows():
        old_name = path_to_images + row["image_id"] + ".jpg"

        if not os.path.isfile(old_name):
            continue

        os.remove(old_name)
        deleted_images += 1

        if deleted_images % 1000 == 0:
            print("Deleted {} images".format(deleted_images))


def rename_images(df: pd.DataFrame, path_to_images: str):
    rename_df = df[df["valid_image"] == True].copy()

    delete_images(df, path_to_images)

    # For every row in the dataframe, rename the image
    for index, row in rename_df.iterrows():
        old_name = path_to_images + row["image_id"] + ".jpg"
        new_name = path_to_images + row["key"]
        
        os.rename(old_name, new_name)

        if index % 10000 == 0:
            print("Renamed {} images".format(index))
            print("Old name: {}".format(old_name))
            print("New name: {}".format(new_name))

def main():

    path_to_images = "../data/images/"
    path_to_df = "../data/multi_label_train_v0.4.pkl"

    df = pd.read_pickle(path_to_df)
    rename_images(df, path_to_images)

    path_to_df = "../data/multi_label_val_v0.4.pkl"
    df = pd.read_pickle(path_to_df)
    rename_images(df, path_to_images)


if __name__ == "__main__":
    main()