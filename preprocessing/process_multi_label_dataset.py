# %%
import pandas as pd
import json

# %%
multi_label_name = "multi_label_train_status_codes"
export_name = "multi_label_train"
version = "v0.4"

# %%
df = pd.read_pickle("../data/"+multi_label_name+".pkl")
df

# %%
if "key" not in df.columns:
    with open("../data/"+export_name+".json") as f:
        data = json.load(f)
    
    keys = data.keys()

    df["key"] = keys

df["key"] = df["key"].apply(lambda x: x.replace("/", "_"))

# %%
if "valid_image" not in df.columns:
    with open("../data/image_status.json") as f:
        data = json.load(f)
    
    df["valid_image"] = df["image_id"].apply(lambda x: data.get(x+".jpg", False))

df

# %%
df["downloadable"] = df["image_id"].apply(lambda x: x.isnumeric() == False and x != "-1")
df

# %%
df["incidents_list"] = df["incidents"].apply(lambda x: ", ".join(sorted([k for k, v in dict(x).items() if v == 1])))
df["incidents_list"] = df["incidents_list"].replace("", "unknown")

df["places_list"] = df["places"].apply(lambda x: ", ".join(sorted([k for k, v in dict(x).items() if v == 1])))
df["places_list"] = df["places_list"].replace("", "unknown")

df

# %%
df.to_pickle("../data/"+export_name+"_"+version+".pkl")

df = pd.read_pickle("../data/"+export_name+"_"+version+".pkl")
df


