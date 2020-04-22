from django.shortcuts import render
import numpy as np
import joblib
import tensorflow as tf
from django.core.files.storage import default_storage

LABELS = ['Bacterial Blight', 'Septorial Brown Spot', 'Frogeye Leaf Spot', 'Healthy', 'Herbicide Injury', 
          'Iron Deficiency Chlorosis', 'Potassium Deficiency', 'Bacterial Pustule', 'Sudden Death Syndrome']

model = tf.keras.models.load_model('.\\model_Monday-09-02-2019-17-36-12')
model_features_extractor = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer("GlobalAveragePooling2D_1").output)
svm_model = joblib.load('.\\svm_rbf_2020-04-21 17_08_26.741869')
# Create your views here.


def predict(request):
    file_upload = request.FILES['image'] 
    img_name = "pic.jpg"
    img_name_saved = default_storage.save(img_name, file_upload)
    img_path = default_storage.url(img_name_saved)
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64, 64))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img_features = model_features_extractor.predict(img)
    result = svm_model.predict(img_features)
    
    return LABELS[result[0]]
