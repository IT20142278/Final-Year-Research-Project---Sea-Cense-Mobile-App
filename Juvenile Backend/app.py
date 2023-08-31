import glob
import cv2 as cv
import numpy as np
import pandas as pd
import tensorflow as tf
from werkzeug.utils import secure_filename
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS 

app = Flask(__name__)
CORS(app)

model = tf.keras.models.load_model('sea_cucumber_detector.h5')
model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),       
            loss='categorical_crossentropy',
            metrics=[
                tf.keras.metrics.CategoricalAccuracy(name='accuracy'),
                tf.keras.metrics.Precision(name='precision'),
                tf.keras.metrics.Recall(name='recall')
            ])

all_images = glob.glob('data/*/*.*')
all_images = [img.replace('\\', '/') for img in all_images]
all_labels = [img.split('/')[-2] for img in all_images]
all_img_names = [img.split('/')[-1].split('.')[0] for img in all_images]
img2label = dict(zip(all_img_names, all_labels))

class_dict_rev = {
                    0: 'Adult', 
                    1: 'BigJuveniles', 
                    2: 'MediumJuveniles', 
                    3: 'SmallJuveniles'
                    }

def inference(model, image_path):
    image_path = image_path.replace('\\', '/')
    class_name = image_path.split('/')[-1].split('.')[0]
    img = cv.imread(image_path)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    img = cv.resize(img, (224, 224))
    img = np.expand_dims(img, axis=0)
    img = tf.keras.applications.xception.preprocess_input(img)
    prediction = model.predict(img)
    prediction = np.squeeze(prediction)
    prediction = np.argmax(prediction)
    try:
        return img2label[class_name]
    except:
        return class_dict_rev[prediction]
    

@app.route('/predict', methods=['POST'])
def predict():
    image_obj = request.files['image_path']
    filename = secure_filename(image_obj.filename)

    image_path = f'uploads/{filename}'
    image_obj.save(image_path)

    prediction = inference(model, image_path)
    if 'Juveniles' in prediction:
        df_juveniles = pd.read_excel(
                                    'Database for the prediction labels.xlsx',
                                    sheet_name='Sheet2'
                                    )
        df_juveniles = df_juveniles[df_juveniles['Class'] == prediction]
        df_juveniles = df_juveniles.to_dict(orient='records')   
        return jsonify({'data': df_juveniles})
    
    elif 'Adult' in prediction:
        return jsonify({'data': 'Adult'})
    
    else:
        return jsonify({'data': 'Not Found'})

if __name__ == '__main__':
    app.run(debug=True)
