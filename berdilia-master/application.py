
from flask import Flask, request, render_template, url_for, make_response, send_from_directory, flash, redirect, jsonify
from werkzeug.utils import secure_filename
import configparser
import uuid
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import numpy as np
from io import BytesIO
import base64
import keras
import os
import os.path
import boto3
import time

config = configparser.ConfigParser()
config.read('conf/application.ini')

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


app_config = config['default']

# Get the DynamoDB service resource.
dynamodb = boto3.resource('dynamodb', region_name=app_config.get('aws_region'))
dynamoDBClient = boto3.client('dynamodb', region_name=app_config.get('aws_region'))

customer_feedback_tbl = dynamodb.Table('comm_utiliateur')

ALLOWED_FILETYPES = set(['.jpg', '.jpeg', '.gif', '.png'])

img_width = app_config.getint('img_width')
img_height = app_config.getint('img_height')

#class_dictionary = np.load(app_config.get('class_dictionary_path'), allow_pickle=True).item()

def load_keras_model(image):
    new_model = keras.models.load_model('mon_model.h5')
    image = np.expand_dims(np.array(image), axis=0)
    predictions_single = new_model.predict(image)
    pred = predictions_single.argmax(axis=1)
    return pred


def get_image_thumbnail(image):
    image.thumbnail((400, 400), resample=Image.LANCZOS)
    image = image.convert("RGB")
    with BytesIO() as buffer:
        image.save(buffer, 'jpeg')
        return base64.b64encode(buffer.getvalue()).decode()

#4
def classify_image(image):
    image = img_to_array(image)


    # important ! sinon les prédictions seront '0'.
    image = image / 255.0

    image = np.expand_dims(image, axis=0)

def index():
    # Gérer la méthode POST de l'envoi
    if request.method == 'POST':
        # Vérifier si la requête Post à le fichier
        if 'bird_image' not in request.files:
            print("[Error] Aucun fichier téléchargé.")
            flash('Aucun fichier téléchargé.')
            return redirect(url_for('index'))

        f = request.files['bird_image']

        if f.filename == '':
            print("[Error] Aucun fichier sélectionné pour le téléchargement.")
            flash('Aucun fichier sélectionné pour le téléchargement.')
            return redirect(url_for('index'))

        sec_filename = secure_filename(f.filename)
        file_extension = os.path.splitext(sec_filename)[1]

        if f and file_extension.lower() in ALLOWED_FILETYPES:
            file_tempname = uuid.uuid4().hex
            image_path = './uploads/' + file_tempname + file_extension
            f.save(image_path)
            file_size = os.path.getsize(image_path)

            file_size_str = str(file_size) + " bytes"
            if (file_size >= 1024):
                if (file_size >= 1024 * 1024):
                    file_size_str = str(file_size // (1024 * 1024)) + " MB"
                else:
                    file_size_str = str(file_size // 1024) + " KB"

            image = load_img(image_path, target_size=(img_width, img_height), interpolation='lanczos')

            orig_image = Image.open(image_path)
            orig_width, orig_height = orig_image.size

            label = classify_image(image=image)
            prediction_probability = classify_image(image=image)

            prediction = load_keras_model(image)

            prediction_probability = 1

            image_data = get_image_thumbnail(image=orig_image)


            sample_data = None
            if prediction == 0:
                label = "Bananaquit"
                sample_image = Image.open('sample_image/Bananaquit.jpg')
                sample_data = get_image_thumbnail(image=sample_image)
            if prediction == 1:
                label = "Black Skimmer"
                sample_image = Image.open('sample_image/BlackSkimmer.jpg')
                sample_data = get_image_thumbnail(image=sample_image)
            if prediction == 2:
                label = "Black Throated Bushtiti"
                sample_image = Image.open('sample_image/BlackThroatedBushtiti.jpg')
                sample_data = get_image_thumbnail(image=sample_image)
            if prediction == 3:
                label = "Cockatoo"
                sample_image = Image.open('sample_image/Cockatoo.jpg')
                sample_data = get_image_thumbnail(image=sample_image)
            if prediction > 3:
                label = "Unknow"
            #os.remove(image_path)

            with application.app_context():
                return render_template('index.html',
                                       label=label,
                                       prob=prediction_probability,
                                       image=image_data,
                                       file_name=sec_filename,
                                       file_size=file_size_str,
                                       sample_image=sample_data,
                                       width=orig_width,
                                       height=orig_height,
                                       )
        else:
            print("[Error] Extension de fichier non autorisée: {}".format(file_extension))
            flash(
                "Le type de fichier que vous avez sélectionné: '{}' pas pris en charge. Veuillez sélectionner a '.jpg', '.jpeg', '.gif', or a '.png' file.".format(
                    file_extension))
            return redirect(url_for('index'))
    else:
        # gérer les méthodes GET, HEAD, et toute autre méthode

        with application.app_context():
            return render_template('index.html' )

def customer_feedback():
    req_json = request.get_json()
    feedback_id = str(uuid.uuid4())
    timestamp = int(time.time())

    feedback = req_json.get('feedback')
    rating = req_json.get('rating')

    if (req_json and feedback and rating):
        try:
            rating = int(rating)

            customer_feedback_tbl.put_item(
                Item={
                    'feedback_id': feedback_id,
                    'timestamp': timestamp,
                    'feedback': feedback,
                    'rating': rating,
                }
            )
        except Exception as e:
            print("[Error] Retour d'erreur : {}".format(e))

    return jsonify(success=True)

def http_413(e):
    print("[Error] Le fichier téléchargé est trop volumineux.")
    flash('Le fichier téléchargé est trop volumineux.')
    return redirect(url_for('index'))

application = Flask(__name__)
application.secret_key = app_config.get('application_secret')

#Ajouter une règle pour la page d'index.
application.add_url_rule('/', 'index', index, methods=['GET', 'POST'])

# AJAX
application.add_url_rule('/feedback', 'feedback', customer_feedback, methods=['POST'])

application.register_error_handler(413, http_413)
application.config['MAX_CONTENT_LENGTH'] = app_config.getint('max_upload_size') * 1024 * 1024


# Exécuter l'application.
if __name__ == "__main__":
    application.debug = app_config.getboolean('debug')
    application.run()