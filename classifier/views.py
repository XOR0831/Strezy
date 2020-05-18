from django.http import JsonResponse
import numpy as np
import joblib
import tensorflow as tf
from django.core.files.storage import default_storage
from django.views.decorators.csrf import csrf_exempt
from PIL import Image

LABELS = ['Bacterial Blight', 'Septorial Brown Spot', 'Frogeye Leaf Spot', 'Healthy', 'Herbicide Injury', 
          'Iron Deficiency Chlorosis', 'Potassium Deficiency', 'Bacterial Pustule', 'Sudden Death Syndrome']

model_binary = tf.keras.models.load_model("./classifier/leaf_final.h5")
model = tf.keras.models.load_model('./classifier/model_Monday-09-02-2019-17-36-12')
model_features_extractor = tf.keras.models.Model(inputs=model.input, outputs=model.get_layer("GlobalAveragePooling2D_1").output)
svm_model = joblib.load('./classifier/svm_rbf_2020-04-21 17_08_26.741869')
# Create your views here.

@csrf_exempt 
def predict(request):
    file_upload = request.FILES['image'] 
    img_name = "pic.jpg"
    img_name_saved = default_storage.save(img_name, file_upload)
    img_path = default_storage.url(img_name_saved)
    img_bin = tf.keras.preprocessing.image.load_img(img_path, target_size=(256, 256))
    img_bin = tf.keras.preprocessing.image.img_to_array(img_bin)
    img_bin = np.expand_dims(img_bin, axis=0)
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64, 64))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    detection = model_binary.predict(img_bin)
    detection = np.argmax(detection, axis=1)
    print(detection)
    if detection == 1:
        img_features = model_features_extractor.predict(img)
        result = svm_model.predict(img_features)

        data = {
            'leaf': True,
            'class': str(LABELS[result[0]])
        }
    else:
        data = {
            'leaf': False,
            'class': None
        }
    
    return JsonResponse(data)
