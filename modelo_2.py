import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2B0, ConvNeXtBase
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import os
from PIL import Image
import json
from pycocotools.coco import COCO
import matplotlib.pyplot as plt
import fasttext.util
fasttext.util.download_model('en', if_exists='ignore')  # English
ft = fasttext.load_model('cc.en.300.bin')
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from pycocoevalcap.cider.cider import Cider
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define a global variable for the number of images to train on
NUM_IMAGES_TO_TRAIN = None  # Use None to indicate that all images should be used

class ImageEncoder(Model):
    def __init__(self, backbone='efficientnetv2', trainable_layers=10):
        super(ImageEncoder, self).__init__()
        if backbone == 'efficientnetv2':
            base_model = EfficientNetV2B0(include_top=False, weights='imagenet', pooling='avg')
        elif backbone == 'convnext':
            base_model = ConvNeXtBase(include_top=False, weights='imagenet', pooling='avg')
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        # Set the last `trainable_layers` layers to be trainable
        for layer in base_model.layers[-trainable_layers:]:
            layer.trainable = True
        self.feature_extractor = base_model

    def call(self, inputs):
        return self.feature_extractor(inputs)  # Extract global features

class TransformerTextDecoder(Model):
    def __init__(self, vocab_size, embed_dim=512, num_heads=8, ff_dim=1024, max_len=50, embedding_matrix=None):
        super(TransformerTextDecoder, self).__init__()
        self.embedding = layers.Embedding(vocab_size, embed_dim, weights=[embedding_matrix], trainable=False)
        self.pos_encoding = self.positional_encoding(max_len, embed_dim)
        self.attention = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = tf.keras.Sequential([
            layers.Dense(ff_dim, activation='relu'),
            layers.Dense(embed_dim),
        ])
        self.layernorm1 = layers.LayerNormalization()
        self.layernorm2 = layers.LayerNormalization()
        self.dropout1 = layers.Dropout(0.1)
        self.dropout2 = layers.Dropout(0.1)
        self.fc = layers.Dense(vocab_size)

    def positional_encoding(self, position, d_model):
        angle_rads = self.get_angles(np.arange(position)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
        return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

    def get_angles(self, pos, i, d_model):
        angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
        return pos * angle_rates

    def call(self, inputs, training=False, return_attention_weights=False):
        text, image_features = inputs
        seq_len = tf.shape(text)[1]
        embedded_text = self.embedding(text) + self.pos_encoding[:, :seq_len, :]
        image_features = tf.tile(image_features, [1, seq_len, 1])
        attn_output, attn_weights = self.attention(query=embedded_text, key=image_features, value=image_features, return_attention_scores=True)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(embedded_text + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        out2 = self.layernorm2(out1 + ffn_output)
        if return_attention_weights:
            return self.fc(out2), attn_weights
        return self.fc(out2)

class BLIPModel(Model):
    def __init__(self, vocab_size, embed_dim=256, num_heads=8, ff_dim=512, max_len=50, backbone='efficientnetv2', embedding_matrix=None):
        super(BLIPModel, self).__init__()
        self.image_encoder = ImageEncoder(backbone=backbone)
        self.text_decoder = TransformerTextDecoder(vocab_size, embed_dim, num_heads, ff_dim, max_len, embedding_matrix)

    def call(self, inputs, training=False):
        images, text_input = inputs
        image_features = self.image_encoder(images)
        text_output = self.text_decoder((text_input, tf.expand_dims(image_features, 1)))
        return text_output


def preprocess_image(image_path, img_size=(224, 224)):
    image = Image.open(image_path).convert('RGB')
    image = image.resize(img_size)
    image = np.array(image) / 255.0
    return image

class ImageCaptionDataset(tf.keras.utils.Sequence):
    def __init__(self, image_dir, captions, tokenizer, batch_size=32, max_len=50):
        self.image_dir = image_dir
        self.captions = captions
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_len = max_len

    def __len__(self):
        return int(np.ceil(len(self.captions) / self.batch_size))

    def __getitem__(self, idx):
        batch_captions = self.captions[idx * self.batch_size:(idx + 1) * self.batch_size]
        images = []
        texts = []
        for image_path, caption in batch_captions:
            images.append(preprocess_image(image_path))
            texts.append(self.tokenizer.tokenizer.texts_to_sequences([caption])[0])
        images = np.array(images)
        texts = pad_sequences(texts, maxlen=self.max_len, padding='post')
        return images, texts


def custom_loss(y_true, y_pred, model):
    # Convert y_true and y_pred to embeddings
    y_true_embedded = model.text_decoder.embedding(y_true)
    y_pred_embedded = tf.matmul(tf.nn.softmax(y_pred, axis=-1), model.text_decoder.embedding.embeddings)

    # Calculate cross-entropy loss
    ce_loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)(y_true, y_pred)

    # Calculate semantic loss using cosine similarity
    semantic_loss = 1 - tf.keras.losses.cosine_similarity(y_true_embedded, y_pred_embedded, axis=-1)

    return ce_loss + 0.1 * semantic_loss

def load_fasttext_embeddings(tokenizer, embed_dim=300):
    vocab_size = len(tokenizer.tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, embed_dim))
    
    for word, i in tokenizer.tokenizer.word_index.items():
        embedding_vector = ft.get_word_vector(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    
    return embedding_matrix

def train_model(model_filepath, tokenizer_filepath, epochs=15, max_len=40):
    # Determine the optimal vocab_size
    num_images = len(set(image_name for image_name, _ in coco_captions))
    print(f"Training on {num_images} images.")
    all_captions = [caption for _, caption in coco_captions]
    tokenizer = CaptionTokenizer()
    tokenizer.fit(all_captions)
    vocab_size = len(tokenizer.tokenizer.word_index) + 1  # +1 for padding token

    # Load FastText embeddings
    embedding_matrix = load_fasttext_embeddings(tokenizer, embed_dim=300)

    # Define the model with the embedding matrix
    model = BLIPModel(
        vocab_size=vocab_size,
        embed_dim=300,  # Ensure this matches the dimension of the FastText embeddings
        num_heads=8,
        ff_dim=512,
        max_len=max_len,
        embedding_matrix=embedding_matrix
    )
    
    batch_size = 32
    # Define the dataset
    image_dir = 'archive/train2017/train2017'
    dataset = ImageCaptionDataset(image_dir=image_dir, captions=coco_captions, tokenizer=tokenizer, batch_size=batch_size, max_len=max_len)

    # Define the optimizer and custom loss function
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
    accuracy = tf.keras.metrics.SparseCategoricalAccuracy()

    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}/{epochs}")
        for batch_index, (images, captions) in enumerate(dataset):
            with tf.GradientTape() as tape:
                outputs = model([images, captions[:, :-1]], training=True)
                total_loss = custom_loss(captions[:, 1:], outputs, model)
                accuracy.update_state(captions[:, 1:], outputs)
            gradients = tape.gradient(total_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
            # Print loss every 10 batches
            if batch_index % batch_size == 0:
                # Compute the mean of the total loss
                mean_loss = tf.reduce_mean(total_loss).numpy()
                print(f"Batch {batch_index}, Loss: {mean_loss:.4f}, Accuracy: {accuracy.result().numpy():.4f}")

        # Save the model weights and tokenizer after each epoch
        epoch_model_filepath = f"{model_filepath}.h5"
        save_model_weights(model, tokenizer, epoch_model_filepath, tokenizer_filepath)

    return model, tokenizer


class CaptionTokenizer:
    def __init__(self, num_words=10000, oov_token="<unk>", max_len=50):
        self.tokenizer = Tokenizer(num_words=num_words, oov_token=oov_token)
        self.max_len = max_len

    def fit(self, captions):
        start_token = '<start>'
        end_token = '<end>'
        processed_captions = [f"{start_token} {caption} {end_token}" for caption in captions]
        self.tokenizer.fit_on_texts(processed_captions)
        
        # Ensure special tokens are in the word index
        if start_token not in self.tokenizer.word_index:
            self.tokenizer.word_index[start_token] = len(self.tokenizer.word_index) + 1
        if end_token not in self.tokenizer.word_index:
            self.tokenizer.word_index[end_token] = len(self.tokenizer.word_index) + 1
        self.tokenizer.word_index['<pad>'] = 0  # Reserve index 0 for padding
        self.tokenizer.index_word[0] = '<pad>'
        
        self.vocab_size = len(self.tokenizer.word_index)

    def encode(self, captions):
        start_token = '<start>'
        end_token = '<end>'
        processed_captions = [f"{start_token} {caption} {end_token}" for caption in captions]
        sequences = self.tokenizer.texts_to_sequences(processed_captions)
        return pad_sequences(sequences, maxlen=self.max_len, padding='post')

    def decode(self, sequence):
        words = [self.tokenizer.index_word.get(idx, '<unk>') for idx in sequence if idx > 0]
        return ' '.join(words).split('<end>')[0]

tokenizer = CaptionTokenizer(num_words=10000, max_len=40)

def load_coco_annotations(annotation_file, image_dir, limit=NUM_IMAGES_TO_TRAIN):
    coco = COCO(annotation_file)
    image_ids = coco.getImgIds()
    captions = []
    count = 0
    for img_id in image_ids:
        if limit is not None and count >= limit:
            break
        img_info = coco.loadImgs(img_id)[0]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        anns = coco.loadAnns(ann_ids)
        for ann in anns:
            image_path = os.path.join(image_dir, img_info['file_name'])
            captions.append((image_path, ann['caption']))
            count += 1
            if limit is not None and count >= limit:
                break
    return captions
annotation_file = 'archive/annotations_trainval2017/annotations/captions_train2017.json'
image_dir = 'archive/train2017/train2017'
coco_captions = load_coco_annotations(annotation_file, image_dir)
dataset = ImageCaptionDataset(
    image_dir=image_dir,
    captions=coco_captions,
    tokenizer=tokenizer,
    batch_size=32,
    max_len=20,
)

def save_model_weights(model, tokenizer, model_filepath, tokenizer_filepath):
    """
    Save the model weights to a file with .h5 extension and the tokenizer to a JSON file.
    """
    # Save model weights
    model.save_weights(model_filepath)
    print(f"Model weights saved to {model_filepath}")

    # Save tokenizer
    tokenizer_json = tokenizer.tokenizer.to_json()
    with open(tokenizer_filepath, 'w') as f:
        f.write(tokenizer_json)
    print(f"Tokenizer saved to {tokenizer_filepath}")

def load_model_weights(model, filepath, sample_input_shape):
    """
    Load the model weights from a file with .h5 extension.
    Ensure the model is built by calling it with a sample input.
    """
    # Build the model by calling it with a sample input
    sample_input = [tf.zeros(sample_input_shape[0]), tf.zeros(sample_input_shape[1])]
    model(sample_input, training=False)
    
    # Load the weights
    model.load_weights(filepath)
    print(f"Model weights loaded from {filepath}")

# Example of building the model and loading weights
def build_and_load_model(model_filepath, tokenizer_filepath, sample_input_shape):
    """
    Build the model by calling it with a sample input and then load the weights and tokenizer.
    """
    # Load the tokenizer
    with open(tokenizer_filepath, 'r') as f:
        tokenizer_json = f.read()
    tokenizer = CaptionTokenizer(max_len=sample_input_shape[1][1])
    tokenizer.tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(tokenizer_json)
    
    # Ensure the tokenizer's vocabulary size is set correctly
    vocab_size = len(tokenizer.tokenizer.word_index) + 1  # +1 for padding token
    print(f"Vocab size: {vocab_size}")

    # Load FastText embeddings
    embedding_matrix = load_fasttext_embeddings(tokenizer, embed_dim=300)

    # Initialize the model with the correct vocabulary size and embedding matrix
    model = BLIPModel(
        vocab_size=vocab_size,
        embed_dim=300,  # Ensure this matches the dimension of the FastText embeddings
        num_heads=8,
        ff_dim=512,
        max_len=sample_input_shape[1][1],
        backbone='efficientnetv2',
        embedding_matrix=embedding_matrix
    )
    
    # Build the model by calling it with a sample input
    sample_image_input = tf.zeros(sample_input_shape[0])  # Shape: (batch_size, height, width, channels)
    sample_text_input = tf.zeros(sample_input_shape[1], dtype=tf.int32)  # Shape: (batch_size, sequence_length)
    model([sample_image_input, sample_text_input], training=False)
    
    # Load the weights
    try:
        model.load_weights(model_filepath)
        print(f"Model weights loaded from {model_filepath}")
    except ValueError as e:
        print(f"Error loading model weights: {e}")
        print("Ensure that the model architecture matches the weights being loaded.")

    print(f"Tokenizer loaded from {tokenizer_filepath}")

    return model, tokenizer

def generate_caption_with_beam_search(model, tokenizer, image_path, max_len=20, beam_width=3):
    # Preprocess the image
    image = preprocess_image(image_path)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Extract image features
    image_features = model.image_encoder(image)

    # Initialize the sequence with the start token
    start_token = tokenizer.tokenizer.word_index['<start>']
    end_token = tokenizer.tokenizer.word_index['<end>']
    sequences = [[list([start_token]), 0.0]]  # List of sequences with their scores

    # Beam search
    for _ in range(max_len):
        all_candidates = []
        for seq, score in sequences:
            if len(seq) > 0 and seq[-1] == end_token:
                all_candidates.append((seq, score))
                continue
            text_input = np.array([seq])  # Add batch and sequence dimensions
            output = model.text_decoder((text_input, tf.expand_dims(image_features, 1)))
            probabilities = tf.nn.softmax(output[:, -1, :], axis=-1).numpy()[0]
            top_k = np.argsort(probabilities)[-beam_width:]
            for word_id in top_k:
                candidate = [seq + [word_id], score - np.log(probabilities[word_id])]
                all_candidates.append(candidate)
        sequences = sorted(all_candidates, key=lambda tup: tup[1])[:beam_width]

    # Choose the sequence with the highest score
    best_sequence = sequences[0][0]
    return tokenizer.decode(best_sequence[1:])  # Exclude the start token


def compare_images(model, tokenizer, image1_path, image2_path, max_len=20):
    # Preprocess the images
    image1 = preprocess_image(image1_path)
    image1 = np.expand_dims(image1, axis=0)  # Add batch dimension
    image2 = preprocess_image(image2_path)
    image2 = np.expand_dims(image2, axis=0)  # Add batch dimension

    # Extract features and calculate differences
    features1 = model.image_encoder(image1)
    features2 = model.image_encoder(image2)
    diff_features = features1 - features2

    # Generate caption based on the difference
    start_token = tokenizer.tokenizer.word_index['<start>']
    text_input = np.array([[start_token]])  # Add batch and sequence dimensions
    generated_caption = []

    for _ in range(max_len):
        output = model.text_decoder((text_input, tf.expand_dims(diff_features, 1)))
        predicted_id = tf.argmax(output[:, -1, :], axis=-1).numpy()[0]
        if predicted_id == tokenizer.tokenizer.word_index['<end>']:
            break
        generated_caption.append(predicted_id)
        text_input = np.append(text_input, [[predicted_id]], axis=1)

    return tokenizer.decode(generated_caption)

# Visualization function
def plot_attention_weights(attention_weights, input_sentence, result, tokenizer):
    fig = plt.figure(figsize=(10, 10))
    for head in range(attention_weights.shape[0]):
        ax = fig.add_subplot(3, 4, head+1)
        ax.matshow(attention_weights[head][0])
        fontdict = {'fontsize': 10}
        ax.set_xticks(range(len(input_sentence)))
        ax.set_yticks(range(len(result)))
        ax.set_xticklabels(input_sentence, fontdict=fontdict, rotation=90)
        ax.set_yticklabels(result, fontdict=fontdict)
        ax.set_xlabel(f'Head {head+1}')
    plt.show()

def evaluate_bleu_score(model, tokenizer, dataset, max_len=20):
    total_bleu_score = 0
    num_samples = 0

    for images, captions in dataset:
        for i in range(len(images)):
            image = images[i:i+1]  # Select a single image
            true_caption = captions[i]
            generated_caption = generate_caption_with_beam_search(model, tokenizer, image, max_len=max_len)
            
            # Decode true and generated captions
            true_caption_decoded = tokenizer.decode(true_caption)
            generated_caption_decoded = tokenizer.decode(generated_caption)
            
            # Calculate BLEU score
            reference = [true_caption_decoded.split()]
            candidate = generated_caption_decoded.split()
            bleu_score = sentence_bleu(reference, candidate)
            total_bleu_score += bleu_score
            num_samples += 1

    average_bleu_score = total_bleu_score / num_samples
    print(f"Average BLEU Score: {average_bleu_score:.4f}")
    return average_bleu_score

def evaluate_rouge_score(model, tokenizer, dataset, max_len=20):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    total_rouge1 = 0
    total_rougeL = 0
    num_samples = 0

    for images, captions in dataset:
        for i in range(len(images)):
            image = images[i:i+1]  # Select a single image
            true_caption = captions[i]
            generated_caption = generate_caption_with_beam_search(model, tokenizer, image, max_len=max_len)
            
            # Decode true and generated captions
            true_caption_decoded = tokenizer.decode(true_caption)
            generated_caption_decoded = tokenizer.decode(generated_caption)
            
            # Calculate ROUGE scores
            scores = scorer.score(true_caption_decoded, generated_caption_decoded)
            total_rouge1 += scores['rouge1'].fmeasure
            total_rougeL += scores['rougeL'].fmeasure
            num_samples += 1

    average_rouge1 = total_rouge1 / num_samples
    average_rougeL = total_rougeL / num_samples
    print(f"Average ROUGE-1 Score: {average_rouge1:.4f}")
    print(f"Average ROUGE-L Score: {average_rougeL:.4f}")
    return average_rouge1, average_rougeL

def evaluate_cider_score(model, tokenizer, dataset, max_len=20):
    cider_scorer = Cider()
    total_cider_score = 0
    num_samples = 0

    for images, captions in dataset:
        for i in range(len(images)):
            image = images[i:i+1]  # Select a single image
            true_caption = captions[i]
            generated_caption = generate_caption_with_beam_search(model, tokenizer, image, max_len=max_len)
            
            # Decode true and generated captions
            true_caption_decoded = tokenizer.decode(true_caption)
            generated_caption_decoded = tokenizer.decode(generated_caption)
            
            # Calculate CIDEr score
            score, _ = cider_scorer.compute_score({0: [true_caption_decoded]}, {0: [generated_caption_decoded]})
            total_cider_score += score
            num_samples += 1

    average_cider_score = total_cider_score / num_samples
    print(f"Average CIDEr Score: {average_cider_score:.4f}")
    return average_cider_score

def augment_images(image_dir, batch_size=32, img_size=(224, 224)):
    datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest'
    )
    generator = datagen.flow_from_directory(
        image_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=None
    )
    return generator
