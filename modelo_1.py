import tensorflow as tf
import os
import json
import pandas as pd
import re
import numpy as np
import time
import matplotlib.pyplot as plt
import collections
import random
import requests
import json
from math import sqrt
from PIL import Image
from tqdm.auto import tqdm
import pickle

# Define o caminho base para os dados
BASE_PATH = 'archive/'

# Carregamento dos dados: Lê o arquivo JSON de anotações e extrai as legendas
with open(f'{BASE_PATH}annotations_trainval2017/annotations/captions_train2017.json', 'r') as f:
    data = json.load(f)
    data = data['annotations']

# Cria pares de nome de imagem e legenda
img_cap_pairs = []

for sample in data:
    img_name = '%012d.jpg' % sample['image_id']
    img_cap_pairs.append([img_name, sample['caption']])

# Converte os pares em um DataFrame e ajusta o caminho das imagens
captions = pd.DataFrame(img_cap_pairs, columns=['image', 'caption'])
captions['image'] = captions['image'].apply(
    lambda x: f'{BASE_PATH}/train2017/train2017/{x}'
)
# Amostra aleatória de 10.000 legendas para processamento
captions = captions.reset_index(drop=True)
captions.head()

# Pré-processamento: Função para limpar e formatar as legendas
def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub('\s+', ' ', text)
    text = text.strip()
    text = '[start] ' + text + ' [end]'
    return text

# Aplica o pré-processamento às legendas
captions['caption'] = captions['caption'].apply(preprocess)
captions.head()

# Hiperparâmetros: Define parâmetros para o modelo de aprendizado
MAX_LENGTH = 40
VOCABULARY_SIZE = 15000
BATCH_SIZE = 32
BUFFER_SIZE = 1000
EMBEDDING_DIM = 512
UNITS = 512
EPOCHS = 15

# Cria e adapta o tokenizador com base nas legendas

def build_tokenizer():
    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=VOCABULARY_SIZE,
        standardize=None,
        output_sequence_length=MAX_LENGTH)

    tokenizer.adapt(captions['caption'])

    # Cria mapeamentos de palavras para índices e vice-versa
    word2idx = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary())

    idx2word = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=tokenizer.get_vocabulary(),
        invert=True)
    
    return tokenizer, word2idx, idx2word

def load_tokenizer(vocab_path):
    with open(vocab_path, 'rb') as vocab_file:
        vocab = pickle.load(vocab_file)

    tokenizer = tf.keras.layers.TextVectorization(
        max_tokens=len(vocab),
        standardize=None,
        output_sequence_length=MAX_LENGTH)
    tokenizer.set_vocabulary(vocab)

    word2idx = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=tokenizer.get_vocabulary())

    idx2word = tf.keras.layers.StringLookup(
        mask_token="",
        vocabulary=tokenizer.get_vocabulary(),
        invert=True)
    
    return tokenizer, word2idx, idx2word

if 'tokenizer' not in globals():
    tokenizer, word2idx, idx2word = build_tokenizer()

# Cria um dicionário que mapeia imagens para suas legendas
img_to_cap_vector = collections.defaultdict(list)
for img, cap in zip(captions['image'], captions['caption']):
    img_to_cap_vector[img].append(cap)

# Embaralha as chaves das imagens e divide em conjuntos de treino e validação
img_keys = list(img_to_cap_vector.keys())
random.shuffle(img_keys)

slice_index = int(len(img_keys)*0.8)
img_name_train_keys, img_name_val_keys = (img_keys[:slice_index], 
                                          img_keys[slice_index:])

# Cria listas de imagens e legendas para treino
train_imgs = []
train_captions = []
for imgt in img_name_train_keys:
    capt_len = len(img_to_cap_vector[imgt])
    train_imgs.extend([imgt] * capt_len)
    train_captions.extend(img_to_cap_vector[imgt])

# Cria listas de imagens e legendas para validação
val_imgs = []
val_captions = []
for imgv in img_name_val_keys:
    capv_len = len(img_to_cap_vector[imgv])
    val_imgs.extend([imgv] * capv_len)
    val_captions.extend(img_to_cap_vector[imgv])

# print(len(train_imgs), len(train_captions), len(val_imgs), len(val_captions))

# Função para carregar e pré-processar dados de imagem e legenda
def load_data(img_path, caption):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    caption = tokenizer(caption)
    return img, caption

# Cria conjuntos de dados de treino e validação
train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_imgs, train_captions))

# Aplica a função de pré-processamento e embaralha os dados
train_dataset = train_dataset.map(
    load_data, num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Cria conjuntos de dados de validação
val_dataset = tf.data.Dataset.from_tensor_slices(
    (val_imgs, val_captions))

# Aplica a função de pré-processamento e embaralha os dados
val_dataset = val_dataset.map(
    load_data, num_parallel_calls=tf.data.AUTOTUNE
    ).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Aplica transformações de aumento de dados às imagens
image_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.2),
        tf.keras.layers.RandomContrast(0.3),
    ]
)

# Define o codificador CNN
def CNN_Encoder():
    inception_v3 = tf.keras.applications.InceptionV3(
        include_top=False,
        weights='imagenet'
    )

    output = inception_v3.output
    output = tf.keras.layers.Reshape(
        (-1, output.shape[-1]))(output)

    cnn_model = tf.keras.models.Model(inception_v3.input, output)
    return cnn_model

# Define a camada do codificador Transformer
class TransformerEncoderLayer(tf.keras.layers.Layer):

    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.layer_norm_1 = tf.keras.layers.LayerNormalization()
        self.layer_norm_2 = tf.keras.layers.LayerNormalization()
        self.attention = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense = tf.keras.layers.Dense(embed_dim, activation="relu")
    
    def call(self, x, training):
        x = self.layer_norm_1(x)
        x = self.dense(x)

        attn_output = self.attention(
            query=x,
            value=x,
            key=x,
            attention_mask=None,
            training=training
        )

        x = self.layer_norm_2(x + attn_output)
        return x
    
# Define a camada de embeddings
class Embeddings(tf.keras.layers.Layer):

    def __init__(self, vocab_size, embed_dim, max_len):
        super().__init__()
        self.token_embeddings = tf.keras.layers.Embedding(
            vocab_size, embed_dim)
        self.position_embeddings = tf.keras.layers.Embedding(
            max_len, embed_dim, input_shape=(None, max_len))


    def call(self, input_ids):
        length = tf.shape(input_ids)[-1]
        position_ids = tf.range(start=0, limit=length, delta=1)
        position_ids = tf.expand_dims(position_ids, axis=0)

        token_embeddings = self.token_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)

        return token_embeddings + position_embeddings
    
# Define a camada do decodificador Transformer
class TransformerDecoderLayer(tf.keras.layers.Layer):

    def __init__(self, embed_dim, units, num_heads):
        super().__init__()
        self.embedding = Embeddings(
            tokenizer.vocabulary_size(), embed_dim, MAX_LENGTH)

        self.attention_1 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )
        self.attention_2 = tf.keras.layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim, dropout=0.1
        )

        self.layernorm_1 = tf.keras.layers.LayerNormalization()
        self.layernorm_2 = tf.keras.layers.LayerNormalization()
        self.layernorm_3 = tf.keras.layers.LayerNormalization()

        self.ffn_layer_1 = tf.keras.layers.Dense(units, activation="relu")
        self.ffn_layer_2 = tf.keras.layers.Dense(embed_dim)

        self.out = tf.keras.layers.Dense(tokenizer.vocabulary_size(), activation="softmax")

        self.dropout_1 = tf.keras.layers.Dropout(0.3)
        self.dropout_2 = tf.keras.layers.Dropout(0.5)
    
    def call(self, input_ids, encoder_output, training, mask=None):
        embeddings = self.embedding(input_ids)

        combined_mask = None
        padding_mask = None
        
        if mask is not None:
            causal_mask = self.get_causal_attention_mask(embeddings)
            padding_mask = tf.cast(mask[:, :, tf.newaxis], dtype=tf.int32)
            combined_mask = tf.cast(mask[:, tf.newaxis, :], dtype=tf.int32)
            combined_mask = tf.minimum(combined_mask, causal_mask)

        # Aplica a primeira camada de atenção
        attn_output_1 = self.attention_1(
            query=embeddings,
            value=embeddings,
            key=embeddings,
            attention_mask=combined_mask,
            training=training
        )

        out_1 = self.layernorm_1(embeddings + attn_output_1)

        # Aplica a segunda camada de atenção
        attn_output_2 = self.attention_2(
            query=out_1,
            value=encoder_output,
            key=encoder_output,
            attention_mask=padding_mask,
            training=training
        )

        out_2 = self.layernorm_2(out_1 + attn_output_2)

        # Aplica a primeira camada da rede neural feed-forward
        ffn_out = self.ffn_layer_1(out_2)
        ffn_out = self.dropout_1(ffn_out, training=training)
        ffn_out = self.ffn_layer_2(ffn_out)

        # Aplica a segunda camada da rede neural feed-forward
        ffn_out = self.layernorm_3(ffn_out + out_2)
        ffn_out = self.dropout_2(ffn_out, training=training)

        # Gera a saída final
        preds = self.out(ffn_out)
        return preds

    # Define a máscara de atenção causal
    def get_causal_attention_mask(self, inputs):
        input_shape = tf.shape(inputs)
        batch_size, sequence_length = input_shape[0], input_shape[1]
        i = tf.range(sequence_length)[:, tf.newaxis]
        j = tf.range(sequence_length)
        mask = tf.cast(i >= j, dtype="int32")
        mask = tf.reshape(mask, (1, input_shape[1], input_shape[1]))
        mult = tf.concat(
            [tf.expand_dims(batch_size, -1), tf.constant([1, 1], dtype=tf.int32)],
            axis=0
        )
        return tf.tile(mask, mult)

# Define o modelo de captionamento de imagem
class ImageCaptioningModel(tf.keras.Model):

    def __init__(self, cnn_model, encoder, decoder, image_aug=None):
        super().__init__()
        self.cnn_model = cnn_model
        self.encoder = encoder
        self.decoder = decoder
        self.image_aug = image_aug
        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.acc_tracker = tf.keras.metrics.Mean(name="accuracy")
    
    # Define a função de cálculo da perda
    def calculate_loss(self, y_true, y_pred, mask):
        loss = self.loss(y_true, y_pred)
        mask = tf.cast(mask, dtype=loss.dtype)
        loss *= mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)

    # Define a função de cálculo da precisão
    def calculate_accuracy(self, y_true, y_pred, mask):
        accuracy = tf.equal(y_true, tf.argmax(y_pred, axis=2))
        accuracy = tf.math.logical_and(mask, accuracy)
        accuracy = tf.cast(accuracy, dtype=tf.float32)
        mask = tf.cast(mask, dtype=tf.float32)
        return tf.reduce_sum(accuracy) / tf.reduce_sum(mask)
    
    # Define a função de cálculo da perda e da precisão
    def compute_loss_and_acc(self, img_embed, captions, training=True):
        encoder_output = self.encoder(img_embed, training=True)
        y_input = captions[:, :-1]
        y_true = captions[:, 1:]
        mask = (y_true != 0)
        y_pred = self.decoder(
            y_input, encoder_output, training=True, mask=mask
        )
        loss = self.calculate_loss(y_true, y_pred, mask)
        acc = self.calculate_accuracy(y_true, y_pred, mask)
        return loss, acc

    # Define a função de treinamento
    def train_step(self, batch):
        imgs, captions = batch

        if self.image_aug:
            imgs = self.image_aug(imgs)
        
        img_embed = self.cnn_model(imgs)

        with tf.GradientTape() as tape:
            loss, acc = self.compute_loss_and_acc(
                img_embed, captions
            )
    
        train_vars = (
            self.encoder.trainable_variables + self.decoder.trainable_variables
        )
        grads = tape.gradient(loss, train_vars)
        self.optimizer.apply_gradients(zip(grads, train_vars))
        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}
    
    # Define a função de teste
    def test_step(self, batch):
        imgs, captions = batch

        img_embed = self.cnn_model(imgs)

        loss, acc = self.compute_loss_and_acc(
            img_embed, captions, training=False
        )

        self.loss_tracker.update_state(loss)
        self.acc_tracker.update_state(acc)

        return {"loss": self.loss_tracker.result(), "acc": self.acc_tracker.result()}

    # Define as métricas
    @property
    def metrics(self):
        return [self.loss_tracker, self.acc_tracker]

    def call(self, inputs, training=False):
        imgs, captions = inputs

        if self.image_aug and training:
            imgs = self.image_aug(imgs)

        img_embed = self.cnn_model(imgs, training=training)
        encoder_output = self.encoder(img_embed, training=training)
        y_pred = self.decoder(captions, encoder_output, training=training)

        return y_pred

def train_model():
    # Instantiate the Transformer encoder and decoder
    encoder = TransformerEncoderLayer(EMBEDDING_DIM, 1)
    decoder = TransformerDecoderLayer(EMBEDDING_DIM, UNITS, 8)

    # Instantiate the image captioning model
    cnn_model = CNN_Encoder()
    caption_model = ImageCaptioningModel(
        cnn_model=cnn_model, encoder=encoder, decoder=decoder, image_aug=image_augmentation,
    )

    # Define the loss function
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction="none"
    )

    # Compile the model
    caption_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=cross_entropy
    )

    # Build the image augmentation model by calling it with a sample input
    if caption_model.image_aug:
        sample_input = tf.random.uniform([1, 299, 299, 3])  # Sample input with shape (batch_size, height, width, channels)
        caption_model.image_aug(sample_input)

    # Explicitly build the model by defining input shapes
    caption_model.build(input_shape=[(None, 299, 299, 3), (None, MAX_LENGTH)])

    # Now you can call summary
    caption_model.summary()

    history = caption_model.fit(
        train_dataset,
        epochs=EPOCHS,
        validation_data=val_dataset
    )

    save_model(caption_model)
    return caption_model

def load_model(model_path):

    # Instantiate the Transformer encoder and decoder
    encoder = TransformerEncoderLayer(EMBEDDING_DIM, 1)
    decoder = TransformerDecoderLayer(EMBEDDING_DIM, UNITS, 8)

    # Instantiate the image captioning model
    cnn_model = CNN_Encoder()
    caption_model = ImageCaptioningModel(
        cnn_model=cnn_model, encoder=encoder, decoder=decoder, image_aug=image_augmentation,
    )

    # Define the loss function
    cross_entropy = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=False, reduction="none"
    )

    # Compile the model
    caption_model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=cross_entropy
    )

    # Build the image augmentation model by calling it with a sample input
    if caption_model.image_aug:
        sample_input = tf.random.uniform([1, 299, 299, 3])  # Sample input with shape (batch_size, height, width, channels)
        caption_model.image_aug(sample_input)

    # Explicitly build the model by defining input shapes
    caption_model.build(input_shape=[(None, 299, 299, 3), (None, MAX_LENGTH)])
    caption_model.summary()
    caption_model.load_weights(model_path)

    return caption_model

# Salva o modelo
def save_model(caption_model):
    # Save the model in TensorFlow SavedModel format
    caption_model.save_weights('model.h5')
    # Save the tokenizer vocabulary
    with open('model/vocab_coco.file', 'wb') as vocab_file:
        pickle.dump(tokenizer.get_vocabulary(), vocab_file)

def load_image_from_path(img_path):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(img)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img


def generate_caption(img_path, add_noise=False, caption_model=None):
    img = load_image_from_path(img_path)
    
    if add_noise:
        noise = tf.random.normal(img.shape)*0.1
        img = img + noise
        img = (img - tf.reduce_min(img))/(tf.reduce_max(img) - tf.reduce_min(img))
    
    img = tf.expand_dims(img, axis=0)
    img_embed = caption_model.cnn_model(img)
    img_encoded = caption_model.encoder(img_embed, training=False)

    y_inp = '[start]'
    for i in range(MAX_LENGTH-1):
        tokenized = tokenizer([y_inp])[:, :-1]
        mask = tf.cast(tokenized != 0, tf.int32)
        pred = caption_model.decoder(
            tokenized, img_encoded, training=False, mask=mask)
        
        pred_idx = np.argmax(pred[0, i, :])
        pred_idx = tf.convert_to_tensor(pred_idx)
        pred_word = idx2word(pred_idx).numpy().decode('utf-8')
        if pred_word == '[end]':
            break
        
        y_inp += ' ' + pred_word
    
    y_inp = y_inp.replace('[start] ', '')
    return y_inp



# idx = random.randrange(0, len(captions))
# img_path = 'archive/val2017/val2017/000000000285.jpg'

# pred_caption = generate_caption(img_path)
# print('Predicted Caption:', pred_caption)
# print()

# img = Image.open(img_path)
# plt.imshow(img)
# plt.axis('off')
# plt.show()