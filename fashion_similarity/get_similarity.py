from keras.models import load_model, Model
import glob
import cv2
import numpy as np
from keras.preprocessing import image
from keras.applications.vgg19 import preprocess_input
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt

model = load_model('./vgg19_2.h5')

model = Model(inputs=model.input,outputs=model.get_layer('flatten_1').output)
model.summary()
img_list = glob.glob('./image/*/*')
logit_list = []
for img_path in img_list:
    img = image.load_img(img_path, target_size=(256, 256))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    logit_list.append(np.array(model.predict(x)).reshape(-1))

knn = NearestNeighbors(n_neighbors=2)
knn.fit(logit_list)

predict = knn.kneighbors(logit_list[-20].reshape(1,-1), return_distance=False)
print(predict)

fig, (ax0, ax1) = plt.subplots(1, 2)
ax0.imshow(cv2.cvtColor(cv2.imread(img_list[predict[0][0]]),cv2.COLOR_BGR2RGB))
ax1.imshow(cv2.cvtColor(cv2.imread(img_list[predict[0][1]]),cv2.COLOR_BGR2RGB))
plt.show()