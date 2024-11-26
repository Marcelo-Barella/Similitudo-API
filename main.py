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
sys.path.append('/modelo_1')
import modelo_1
sys.path.append('/modelo_2')
import modelo_2

MAX_LENGTH = 40
VOCABULARY_SIZE = 15000
BATCH_SIZE = 64
BUFFER_SIZE = 1000
EMBEDDING_DIM = 512
UNITS = 512
EPOCHS = 15
# Carrega o modelo salvo
# caption_model = modelo_1.train_model()
caption_model = modelo_1.load_model('model/model.h5')
caption_model, tokenizer = modelo_2.build_and_load_model('model.h5', 'tokenizer.json', sample_input_shape=[(1, 224, 224, 3), (1, MAX_LENGTH)])

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
        pred_caption = modelo_1.generate_caption(img_path=file_path, caption_model=caption_model)
        translator = Translator()
        translated_caption = translator.translate(pred_caption, src='en', dest='pt').text
        translated_caption = translated_caption.capitalize()
        return {
            "message": f"Image {filename} uploaded successfully",
            "caption": translated_caption
        }, 200

class DoubleImageUpload(Resource):
    def post(self):
        if 'image1' not in request.files or 'image2' not in request.files:
            return {"message": "Both images are required"}, 400
        
        file1 = request.files['image1']
        file2 = request.files['image2']
        
        if file1.filename == '' or file2.filename == '':
            return {"message": "No selected file"}, 400
        
        # Read the binary data from the files
        image_data1 = file1.read()
        image_data2 = file2.read()
        
        # Save the binary data to files
        filename1 = secure_filename(file1.filename)
        filename2 = secure_filename(file2.filename)
        file_path1 = os.path.join(app.config['UPLOAD_FOLDER'], filename1)
        file_path2 = os.path.join(app.config['UPLOAD_FOLDER'], filename2)
        
        with open(file_path1, 'wb') as f:
            f.write(image_data1)
        with open(file_path2, 'wb') as f:
            f.write(image_data2)
        
        # Generate captions for both images
        caption1 = modelo_1.generate_caption(img_path=file_path1, caption_model=caption_model)
        caption2 = modelo_1.generate_caption(img_path=file_path2, caption_model=caption_model)
        
        # Compare the images and generate a caption based on the differences
        pred_caption = modelo_2.compare_images(caption_model, tokenizer, image1_path=file_path1, image2_path=file_path2, max_len=MAX_LENGTH)
        
        return {
            "message": f"Images {filename1} and {filename2} uploaded successfully",
            "caption": pred_caption,
            "caption1": caption1,
            "caption2": caption2,
        }, 200

api.add_resource(SingleImageUpload, '/interpretacao')
api.add_resource(DoubleImageUpload, '/comparacao')

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')
