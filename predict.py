import numpy as np
import random
import matplotlib.pyplot as plt
import os
from utils.model import load_model
import valohai

# Define that data and model paths
#path = valohai.inputs("test_dataset").paths()
model_paths_all = valohai.inputs('model').paths()
dataset_names = valohai.parameters('dataset_names').value

# Possible ship categories
category = {'Cargo': 1, 
'Military': 2, 
'Carrier': 3, 
'Cruise': 4, 
'Tankers': 5}

for dataset in dataset_names:
    path = valohai.inputs('test_dataset').path('test/'+dataset+'/*')
    print(path)

    # Run predictions for all models provided as inputs
    for model_path in model_paths_all:
        if dataset in model_path:
            print(model_path)
            model = load_model(model_path)
            head, tail = os.path.split(model_path)
            model_name = tail.rstrip(".h5")

    with np.load(path, allow_pickle=True) as f:
        test_data = f['test_data']

    predictions = model.predict(test_data)

    # Pick 3 random images from test set to save with the predicted cateogory
    test_img = []
    for i in range(0, 3):
        y = random.randrange(len(test_data))
        test_img.append(y)

    # Save images and predictions
    for i in test_img: 
        plt.imshow(test_data[i])
        
        for key in category:
            if category[key] == np.argmax(predictions[i])+1:
                im_path = "predictions/"+model_name+"/" + "img" + str(i) + "_" + key
                plt.savefig(valohai.outputs().path(im_path))