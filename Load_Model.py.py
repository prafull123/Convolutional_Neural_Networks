from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json,model_from_yaml
import numpy

# load YAML and create model
yaml_file = open('CNN_model.yaml', 'r')
loaded_model_yaml = yaml_file.read()


classifier = model_from_yaml(loaded_model_yaml)
# load weights into new model
classifier.load_weights("CNN_model.h5")
print("Loaded model from disk")


# Making new predictions

import numpy as np
from keras.preprocessing import image
import matplotlib.pyplot as plt

test_image = image.load_img('Road_2.jpg', target_size = (64, 64))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)

# make a prediction
result = classifier.predict_classes(test_image)
# show the inputs and predicted outputs
print ("RESULT---")
if (result[0][0]) == 1:
    print ("Its a ROAD")
else:
    print ("Its a FIELD")


import matplotlib.image as mpimg
import numpy as np

img=mpimg.imread('Road_2.jpg')
imgplot = plt.imshow(img)



yaml_file.close()


