import os
import numpy as np
from flask import Flask, request
from flask_restful import Api, Resource
from werkzeug.utils import secure_filename
from googletrans import Translator
from flask_cors import CORS

# Load the trained model
import tensorflow as tf
import sys
sys.path.append('/kaggle-transformer')
import kaggle_transformer

MAX_LENGTH = 40
VOCABULARY_SIZE = 15000
BATCH_SIZE = 64
BUFFER_SIZE = 1000
EMBEDDING_DIM = 512
UNITS = 512
EPOCHS = 15
# Carrega o modelo salvo
# caption_model = kaggle_transformer.train_model()
caption_model = kaggle_transformer.load_model('model/model.h5')

app = Flask(__name__)
CORS(app)
api = Api(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

class SingleImageUpload(Resource):
    def post(self):
        if 'image' not in request.files:
            return {"message": "No image part"}, 400
        
        file = request.files['image']
        
        if file.filename == '':
            return {"message": "No selected file"}, 400
        
        # Read the binary data from the file
        image_data = file.read()
        
        # Save the binary data to a file
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(file_path, 'wb') as f:
            f.write(image_data)
        pred_caption = kaggle_transformer.generate_caption(img_path=file_path, caption_model=caption_model)
        translator = Translator()
        translated_caption = translator.translate(pred_caption, src='en', dest='pt').text
        
        return {
            "message": f"Image {filename} uploaded successfully",
            "caption": translated_caption
        }, 200

class DoubleImageUpload(Resource):
    def post(self):
        # Leave this method blank as requested
        pass

api.add_resource(SingleImageUpload, '/interpretacao')
api.add_resource(DoubleImageUpload, '/comparacao')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
