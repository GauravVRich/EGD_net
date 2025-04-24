# EGDnet
Edge Guided Network for Monocular depth estimation

# Instructions
1.clone the repository
2.run `pip install -r requirements.txt`

# Download dataset 
download dataset for this link:
`https://drive.google.com/file/d/1SzqljaIwlyAQJFW1ke0rsP3g_GRouZyT/view?usp=drivesdk`

The dataset contains images,depth(.npy),edges and gradient for three different environments-Pillar World, Urban and Downtown.

# Project structure 


root_folder/

|--datasets/(**folders and sub-folders of the dataset)

|--train.py

|--generate_maps.py


## To train the model
Run `python train.py`
