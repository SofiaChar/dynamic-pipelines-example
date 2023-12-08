import cv2
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from utils.image import ImagePreprocessing
import warnings
warnings.filterwarnings('ignore')
import valohai

# Get data paths
rcsv= pd.read_csv(valohai.inputs("labels").path())
data_path = valohai.inputs("dataset").paths()

data=[]
for file in data_path:
    image = cv2.imread(file)
    data.append(image)

if valohai.parameters('dataset_name').value ==  "train":
    train_images = data[:6252]
    test_images= data[6252:]
elif valohai.parameters('dataset_name').value ==  "train_harbor_A":
    train_images = data[:2084]
    test_images= data[6252:7145]
elif valohai.parameters('dataset_name').value ==  "train_harbor_B":
    train_images = data[2085:4169]
    test_images= data[7146:8041]
elif valohai.parameters('dataset_name').value ==  "train_harbor_C":
    train_images = data[4170:6252]
    test_images= data[8041:]
else:
    print("Invalid dataset name.")

preprocess = ImagePreprocessing(train_images , test_images , height=150 , length= len(data) , dataframe=rcsv)
rez_images , LABELS , test_rez_images = preprocess.Reshape()
onehot_labels = preprocess.OneHot(LABELS)
X_train , X_val , Y_train , Y_val = preprocess.splitdata(rez_images , onehot_labels )

print('Saving preprocessed data...')
# Save training data
path = valohai.outputs().path('preprocessed_data.npz')
np.savez_compressed(path, x_train=X_train, y_train=Y_train, x_val=X_val, y_val=Y_val)

# Save preprocessed test images
path_test_data = valohai.outputs("test").path('preprocessed_test_data.npz')
np.savez_compressed(path_test_data, test_data=test_rez_images)
print('Save completed')