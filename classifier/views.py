from django.http import JsonResponse
import numpy as np
import joblib
import tensorflow as tf
from django.core.files.storage import default_storage
from django.views.decorators.csrf import csrf_exempt
from PIL import Image
from .models import History

LABELS = ['Bacterial Blight', 'Septorial Brown Spot', 'Frogeye Leaf Spot', 'Healthy', 'Herbicide Injury', 
          'Iron Deficiency Chlorosis', 'Potassium Deficiency', 'Bacterial Pustule', 'Sudden Death Syndrome']
BIOTIC = ['Bacterial Blight', 'Bacterial Pustule', 'Sudden Death Syndrome', 'Septorial Brown Spot', 'Frogeye Leaf Spot']
DESCRIPTION = ['Lesions are angular shaped with reddish brown centers. Margins of disease spot surrounded by yellow halos. Lesions coalesce into larger irregularly shaped area that fall out.',
               'Lesions are irregular-shaped and dark brown in color. Adjacent lesions grow together and form larger blotches which are darker than the lesions of other diseases',
               'Lesions are circular with gray to light brown center with dark reddish-brown margins. Lesions diameter range from 1-5 mm.',
               'Your Plant is Healthy!',
               'Small brown specking to severe bronzing and bleaching.',
               'Yellowing of interveinal areas of leaves, while the veins remain green. Later on, small brown specks on affected leaves.',
               'Continuous yellow margin at the tip of the leaf. Yellowing is followed by necrosis',
               'Lesions are small and surrounded by yellow halo and some lesions have pin-point brown spots. Small pustule lesions are located on the underside of the leaf.',
               'Large irregularly shaped blotches of necrosis between the veins.']

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
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64, 64))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    detection = int(model_binary.predict(img)[0][0])
    print(detection)
    if detection == 1:
        img_features = model_features_extractor.predict(img)
        result = svm_model.predict(img_features)
        if LABELS[result[0]] in BIOTIC:
            types = "Biotic Stress"
        else:
            types = "Abiotic Stress"
        history = History(
            title=str(LABELS[result[0]]),
            types=types,
            description=str(DESCRIPTION[result[0]])
        )
        history.save()
        data = {
            'leaf': True,
            'types': types,
            'description': str(DESCRIPTION[result[0]]),
            'class': str(LABELS[result[0]])
        }
    else:
        data = {
            'leaf': False,
            'types': None,
            'description': None,
            'class': None
        }
    
    return JsonResponse(data)


@csrf_exempt 
def predict_class_only(request):
    file_upload = request.FILES['image'] 
    img_name = "pic.jpg"
    img_name_saved = default_storage.save(img_name, file_upload)
    img_path = default_storage.url(img_name_saved)
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=(64, 64))
    img = tf.keras.preprocessing.image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img_features = model_features_extractor.predict(img)
    result = svm_model.predict(img_features)
    if LABELS[result[0]] in BIOTIC:
        types = "Biotic Stress"
    else:
        types = "Abiotic Stress"
    history = History(
        title=str(LABELS[result[0]]),
        types=types,
        description=str(DESCRIPTION[result[0]])
    )
    history.save()
    data = {
        'leaf': True,
        'types': types,
        'description': str(DESCRIPTION[result[0]]),
        'class': str(LABELS[result[0]])
    }
    
    return JsonResponse(data)


@csrf_exempt
def show_history(request):
    history = History.objects.order_by('-datetime').values('id', 'title', 'types', 'description','datetime')
    data = {
        'history': list(history)
    }
    return JsonResponse(data)
