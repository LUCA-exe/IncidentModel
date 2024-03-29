# Real-time incident detection



# Setup environment

```
git clone https://github.com/ethanweber/IncidentsDataset
cd IncidentsDataset

conda create -n incidents python=3.8.2
conda activate incidents
pip install -r requirements.txt
```

# Using the Incident Model

1. Download pretrained weights [here](https://drive.google.com/drive/folders/1k2nggK3LqyBE5huGpL3E-JXoEv7o6qRq?usp=sharing). Place desired files in the [pretrained_weights](pretrained_weights/) folder. Note that these take the following structure:

   ```
   # run this script to download everything
   python run_download_weights.py

   # pretrained weights with Places 365
   resnet18_places365.pth.tar
   resnet50_places365.pth.tar

   # ECCV baseline model weights
   eccv_baseline_model_trunk.pth.tar
   eccv_baseline_model_incident.pth.tar
   eccv_baseline_model_place.pth.tar

   # ECCV final model weights
   eccv_final_model_trunk.pth.tar
   eccv_final_model_incident.pth.tar
   eccv_final_model_place.pth.tar

   # multi-label final model weights
   multi_label_final_model_trunk.pth.tar
   multi_label_final_model_incident.pth.tar
   multi_label_final_model_place.pth.tar
   ```

2. Run inference with the model with [RunModel.ipynb](RunModel.ipynb).

3. Compute mAP and report numbers.

   ```
   # test the model on the validation set
   python run_model.py \
       --config=configs/eccv_final_model \
       --mode=val \
       --checkpoint_path=pretrained_weights \
       --images_path=/path/to/downloaded/images/folder/
   ```

4. Train a model.

   ```
   # train the model
   python run_model.py \
       --config=configs/eccv_final_model \
       --mode=train \
       --checkpoint_path=runs/eccv_final_model

   # visualize tensorboard
   tensorboard --samples_per_plugin scalars=100,images=10 --port 8880 --bind_all --logdir runs/eccv_final_model
   ```

   See the `configs/` folder for more details.

# (Inspired from) Incidents1M: a large-scale dataset of images with natural disasters, damage, and incidents

See the following pages for more details:

- Project page: [http://incidentsdataset.csail.mit.edu/](http://incidentsdataset.csail.mit.edu/) or [https://ethanweber.me/IncidentsDataset](https://ethanweber.me/IncidentsDataset).
- ECCV 2020 Paper "Detecting natural disasters, damage, and incidents in the wild" [here](https://arxiv.org/abs/2008.09188).
- **Extended Paper** "Incidents1M: a large-scale dataset of images with natural disasters, damage, and incidents" [here](https://arxiv.org/abs/2201.04236).

If you find this work helpful for your research, please consider citing our paper:

```
@InProceedings{weber2020eccv,
  title={Detecting natural disasters, damage, and incidents in the wild},
  author={Weber, Ethan and Marzo, Nuria and Papadopoulos, Dim P. and Biswas, Aritro and Lapedriza, Agata and Ofli, Ferda and Imran, Muhammad and Torralba, Antonio},
  booktitle={The European Conference on Computer Vision (ECCV)},
  month = {August},
  year={2020}
}
```

# Original repository

```
git clone https://github.com/ethanweber/IncidentsDataset
```
