import cv2
import numpy as np 
import pandas as pd
from utils.image import ImagePreprocessing
import valohai
import json

# Get data paths
data_path = valohai.inputs("dataset").paths()
dataset_names = valohai.parameters('dataset_names').value

data=[]
for file in data_path:
    image = cv2.imread(file)
    data.append(image)

images = {
    'all_harbors': [data[:6252],data[6252:]],
    'harbor_A': [data[:2084],data[6252:7145]],
    'harbor_B': [data[2085:4169], data[7146:8041]],
    'harbor_C': [data[4170:6252], data[8041:]]
}

# Read the execution details from the configuration file for dataset naming
f = open('/valohai/config/execution.json')
exec_details = json.load(f)

# Get the execution ID
exec_id = exec_details['valohai.execution-id']

# Preprocess the data

for dataset in dataset_names:
    train_images = images[dataset][0]
    test_images = images[dataset][1]
    
    # Get the correct labels
    rcsv= pd.read_csv(valohai.inputs("labels").path('train_' + dataset + '.csv'))

    preprocess = ImagePreprocessing(train_images , test_images , height=150 , length=len(train_images) , dataframe=rcsv)
    rez_images , LABELS , test_rez_images = preprocess.Reshape()
    onehot_labels = preprocess.OneHot(LABELS)
    X_train , X_val , Y_train , Y_val = preprocess.splitdata(rez_images , onehot_labels )

    print('Saving preprocessed data...')
    # Save preprocessed training data
    path = valohai.outputs('train').path('preprocessed_data_' + dataset +'.npz')
    np.savez_compressed(path, x_train=X_train, y_train=Y_train, x_val=X_val, y_val=Y_val)

    metadata_train = {
        'valohai.dataset-versions': ['dataset://'+ dataset + '_train/' + exec_id]
    }
    
    metadata_path = valohai.outputs('train').path('preprocessed_data_' + dataset +'.npz.metadata.json')
    with open(metadata_path, 'w') as outfile:
        json.dump(metadata_train, outfile)

    # Save preprocessed test data
    path_test_data = valohai.outputs('test').path('preprocessed_test_data_' + dataset +'.npz')
    np.savez_compressed(path_test_data, test_data=test_rez_images)

    metadata_test = {
        'valohai.dataset-versions': ['dataset://'+ dataset + '_test/'+ exec_id]
    }
    
    metadata_path = valohai.outputs('test').path('preprocessed_test_data_' + dataset +'.npz.metadata.json')
    with open(metadata_path, 'w') as outfile:
        json.dump(metadata_test, outfile)


    print('Save completed')

print(json.dumps({"run_training_for": dataset_names}))
