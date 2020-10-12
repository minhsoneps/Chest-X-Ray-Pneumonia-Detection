# Chest-X-Ray-Pneumonia-Detection
To run this model on a Linux machine:

Download the dataset from this link (file ChestXRay2017.zip):
https://data.mendeley.com/datasets/rscbjbr9sj/2/files/f12eaf6d-6023-432f-acc9-80c9d7393433 

Unzip the data to the folder data/ inside folder Final Project/ with folders organized as follows:
Final Project/
   data/
      test/
         NORMAL/
         PNEUMONIA/
      train/
         NORMAL/
         PNEUMONIA/

It could be required to install some packages to this environment by running these commands:

pip install opencv-python

pip install --upgrade pip

conda install -c conda-forge keras

After successfully installing these packages, run the command: 
python final_project.py
