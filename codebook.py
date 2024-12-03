# https://github.com/angeligareta/image-captioning/blob/master/notebooks/image-captioning.ipynb is taken as reference for our work.
# I tried to use the Vision transformers for extracting the features from the images.

# Dependencies
import re
import random

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow import keras
from time import time

from tqdm import tqdm # progress bar
from sklearn.model_selection import train_test_split # Dividing train test
from nltk.translate.bleu_score import corpus_bleu # BLEU Score

# Dataset
dataset_path = "/Users/vikasvicky/PycharmProjects/pythonProject/ML_2/Assignments/Assignment_5/Flickr8K"
dataset_images_path = dataset_path + "/Images/"

# Images configuration
img_height = 180
img_width = 180
validation_split = 0.2

# ViT Feature Extractor and Model
feature_extractor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
vit_model = TFViTForImageClassification.from_pretrained('google/vit-base-patch16-224-in21k')
###################################

# Initialize the encoder with the ViT model
encoder = get_encoder()

##################
# Read captions
# Preprocess the caption, splitting the string and adding <start> and <end> tokens
def get_preprocessed_caption(caption):
    caption = re.sub(r'\s+', ' ', caption)
    caption = caption.strip()
    caption = "<start> " + caption + " <end>"
    return caption
##################
images_captions_dict = {}

with open(dataset_path + "/captions.txt", "r") as dataset_info:
    next(dataset_info) # Omit header: image, caption

    # Using a subset of 4,000 entries out of 40,000
    for info_raw in list(dataset_info)[:4000]:
        info = info_raw.split(",")
        image_filename = info[0]
        caption = get_preprocessed_caption(info[1])

        if image_filename not in images_captions_dict.keys():
            images_captions_dict[image_filename] = [caption]
        else:
            images_captions_dict[image_filename].append(caption)
##################
# Read images
# Create dictionary with image filename as key and the image feature extracted using the pretrained model as the value.
def load_image(image_path):
    img = tf.io.read_file(dataset_images_path + image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (img_height, img_width))
    img = tf.keras.applications.inception_v3.preprocess_input(img) # preprocessing needed for pre-trained model
    return img, image_path
##################
image_captions_dict_keys = list(images_captions_dict.keys())
image_dataset = tf.data.Dataset.from_tensor_slices(image_captions_dict_keys)
image_dataset = image_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(64)
##################
images_dict = {}
encoder = get_encoder()
for img_tensor, path_tensor in tqdm(image_dataset):
    batch_features_tensor = encoder(img_tensor)

    # Loop over batch to save each element in images_dict
    for batch_features, path in zip(batch_features_tensor, path_tensor):
        decoded_path = path.numpy().decode("utf-8")
        images_dict[decoded_path] = batch_features.numpy()
##################
list(images_dict.items())[0][1].shape
##################
# Display image from original dataset
plt.imshow(load_image('1000268201_693b08cb0e.jpg')[0].numpy())
#################
# Get images and labels from filenames
def get_images_labels(image_filenames):
    images = []
    labels = []

    for image_filename in image_filenames:
        image = images_dict[image_filename]
        captions = images_captions_dict[image_filename]

        # Add one instance per caption
        for caption in captions:
            images.append(image)
            labels.append(caption)

    return images, labels

#Generate train and test set
#This approach divides image_filenames, to avoid same image with different caption in train and test dataset.
# Also the resulting train test is not shuffled because a tensorflow native method will be used for that aim.
image_filenames = list(images_captions_dict.keys())
image_filenames_train, image_filenames_test = \
    train_test_split(image_filenames, test_size=validation_split, random_state=1)

X_train, y_train_raw = get_images_labels(image_filenames_train)
X_test, y_test_raw = get_images_labels(image_filenames_test)

# Per image 5 captions and 0.2 test split
len(X_train), len(y_train_raw), len(X_test), len(y_test_raw)
##################
# Tokenize train labels
# Generate a vocabulary and transform the train captions to a vector with their indices in the vocabulary [1].
top_k = 5000 # Take maximum of words out of 7600
tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=top_k,
                                                  oov_token="<unk>",
                                                  filters='!"#$%&()*+.,-/:;=?@[\]^_`{|}~ ')

# Generate vocabulary from train captions
tokenizer.fit_on_texts(y_train_raw)

# Introduce padding to make the captions of the same size for the LSTM model
tokenizer.word_index['<pad>'] = 0
tokenizer.index_word[0] = '<pad>'

# Create the tokenized vectors
y_train = tokenizer.texts_to_sequences(y_train_raw)

# Add padding to each vector to the max_length of the captions (automatically done)
y_train = tf.keras.preprocessing.sequence.pad_sequences(y_train, padding='post')
##################
max_caption_length = max(len(t) for t in y_train)
print(max_caption_length)
##################
# Example
[tokenizer.index_word[i] for i in y_train[1]]
##################
# Generate Tensorflow dataset
# Generate dataset using buffer and batch size that would be used during training.
dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
BUFFER_SIZE = len(X_train)
BATCH_SIZE = 64
NUM_STEPS = BUFFER_SIZE // BATCH_SIZE

# Shuffle and batch
dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

# Using prefetching: https://www.tensorflow.org/guide/data_performance#prefetching
dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
##################
########### /// Models_Definition /// #########
# CNN Encoder
class CNN_Encoder(tf.keras.Model):
    # Since you have already extracted the features and dumped it using pickle
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.flat = tf.keras.layers.Flatten()
        self.fc = tf.keras.layers.Dense(embedding_dim) #, activation='relu')

    def call(self, x):
        x = self.flat(x)
        x = self.fc(x)
        return x

# Select a random image from the test set
test_img_name = random.choice(image_filenames_test)

# Load the image
raw_img = load_image(test_img_name)[0]

# Display the image
plt.imshow(raw_img)
plt.axis('off')
plt.show()

# Print the actual captions
print("Real captions:")
for caption in images_captions_dict[test_img_name]:
    print(caption)

################
# /// LSTM Decoder ///
class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        # input_dim = size of the vocabulary
        # Define the embedding layer to transform the input caption sequence
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # Define the Long Short Term Memory layer to predict the next words in the sequence
        self.lstm = tf.keras.layers.LSTM(self.units, return_sequences=True, return_state=True)

        # Define a dense layer to transform the LSTM output into prediction of the best word
        self.fc = tf.keras.layers.Dense(vocab_size)  # , activation='softmax')

    # A function that transforms the input embeddings and passes them to the LSTM layer
    def call(self, captions, features, omit_features=False, initial_state=None, verbose=False):
        if verbose:
            print("Before embedding")
            print(captions.shape)

        embed = self.embedding(captions)  # (batch_size, 1, embedding_dim)

        if verbose:
            print("Embed")
            print(embed.shape)

        features = tf.expand_dims(features, 1)

        if verbose:
            print("Features")
            print(features.shape)

        # Concatenating the image and caption embeddings before providing them to LSTM
        # shape == (batch_size, 1, embedding_dim + hidden_size)
        lstm_input = tf.concat([features, embed], axis=-2) if (omit_features == False) else embed

        if verbose:
            print("LSTM input")
            print(lstm_input.shape)

        # Passing the concatenated vector to the LSTM
        output, memory_state, carry_state = self.lstm(lstm_input, initial_state=initial_state)

        if verbose:
            print("LSTM output")
            print(output.shape)

        # Transform LSTM output units to vocab_size
        output = self.fc(output)

        return output, memory_state, carry_state

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
#################### /// SIMPLE RNN/// ########################################################################
class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        # input_dim = size of the vocabulary
        # Define the embedding layer to transform the input caption sequence
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # Define the Simple RNN layer to predict the next words in the sequence
        self.rnn = tf.keras.layers.SimpleRNN(self.units, return_sequences=True, return_state=True)

        # Define a dense layer to transform the RNN output into prediction of the best word
        self.fc = tf.keras.layers.Dense(vocab_size)

    # A function that transforms the input embeddings and passes them to the RNN layer
    def call(self, captions, features, omit_features=False, initial_state=None, verbose=False):
        if verbose:
            print("Before embedding")
            print(captions.shape)

        embed = self.embedding(captions)  # (batch_size, 1, embedding_dim)

        if verbose:
            print("Embed")
            print(embed.shape)

        features = tf.expand_dims(features, 1)

        if verbose:
            print("Features")
            print(features.shape)

        # Concatenating the image and caption embeddings before providing them to the RNN
        # shape == (batch_size, 1, embedding_dim + hidden_size)
        rnn_input = tf.concat([features, embed], axis=-2) if (omit_features == False) else embed

        if verbose:
            print("RNN input")
            print(rnn_input.shape)

        # Passing the concatenated vector to the RNN
        output, state = self.rnn(rnn_input, initial_state=initial_state)

        if verbose:
            print("RNN output")
            print(output.shape)

        # Transform RNN output units to vocab_size
        output = self.fc(output)

        return output, state

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))
#################################### /// GRU Decoder ///  ####################################
class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units

        # input_dim = size of the vocabulary
        # Define the embedding layer to transform the input caption sequence
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)

        # Define the GRU layer to predict the next words in the sequence
        self.gru = tf.keras.layers.GRU(self.units, return_sequences=True, return_state=True)

        # Define a dense layer to transform the GRU output into prediction of the best word
        self.fc = tf.keras.layers.Dense(vocab_size)

    # A function that transforms the input embeddings and passes them to the GRU layer
    def call(self, captions, features, omit_features=False, initial_state=None, verbose=False):
        if verbose:
            print("Before embedding")
            print(captions.shape)

        embed = self.embedding(captions)  # (batch_size, 1, embedding_dim)

        if verbose:
            print("Embed")
            print(embed.shape)

        features = tf.expand_dims(features, 1)

        if verbose:
            print("Features")
            print(features.shape)

        # Concatenating the image and caption embeddings before providing them to the GRU
        # shape == (batch_size, 1, embedding_dim + hidden_size)
        gru_input = tf.concat([features, embed], axis=-2) if (omit_features == False) else embed

        if verbose:
            print("GRU input")
            print(gru_input.shape)

        # Passing the concatenated vector to the GRU
        output, state = self.gru(gru_input, initial_state=initial_state)

        if verbose:
            print("GRU output")
            print(output.shape)

        # Transform GRU output units to vocab_size
        output = self.fc(output)

        return output, state

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))


##############################################################################################################################
##################
# Train Stage

units = embedding_dim = 512  # As in the paper
vocab_size = min(top_k + 1, len(tokenizer.word_index.keys()))

# Initialize encoder and decoder
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

# Initialize optimizer
optimizer = tf.keras.optimizers.legacy.Adam()

# As the label is not one-hot encoded but indices. Logits as they are not probabilities.
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')


# Computes the loss using SCCE and calculates the average of singular losses in the tensor
def loss_function(real, pred, verbose=False):
    loss_ = loss_object(real, pred)

    if verbose:
        print("Loss")
        print(loss_)

    loss_ = tf.reduce_mean(loss_, axis=1)

    if verbose:
        print("After Mean Axis 1")
        print(loss_)

    return loss_

#################################### /// LSTM train step /// ############################################################
@tf.function
def train_step(img_tensor, target, verbose=False):
    if verbose:
        print("Image tensor")
        print(img_tensor.shape)

        print("Target")
        print(target.shape)

        # The input would be each set of words without the last one (<end>), to leave space for the first one that
    # would be the image embedding
    dec_input = tf.convert_to_tensor(target[:, :-1])

    # Source: https://www.tensorflow.org/api_docs/python/tf/GradientTape
    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        if verbose:
            print("Features CNN")
            print(features)

        predictions, _, _ = decoder(dec_input, features, verbose=verbose)

        if verbose:
            print("Predictions RNN")
            print(predictions)

        caption_loss = loss_function(target, predictions)  # (batch_size, )

        # After tape
        total_batch_loss = tf.reduce_sum(caption_loss)  # Sum (batch_size, ) => K
        mean_batch_loss = tf.reduce_mean(caption_loss)  # Mean(batch_size, ) => K

    # Updated the variables
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(caption_loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return total_batch_loss, mean_batch_loss
################## /// Simple RNN train-step function/// ##################
@tf.function
def train_step(img_tensor, target, verbose=False):
    if verbose:
        print("Image tensor")
        print(img_tensor.shape)

    # The input would be each set of words without the last one (<end>), to leave space for the first one that
    # would be the image embedding
    dec_input = tf.convert_to_tensor(target[:, :-1])

    # Source: https://www.tensorflow.org/api_docs/python/tf/GradientTape
    with tf.GradientTape() as tape:
        features = encoder(img_tensor)

        if verbose:
            print("Features CNN")
            print(features)

        predictions, _ = decoder(dec_input, features, verbose=verbose)  # Updated to unpack only two values

        if verbose:
            print("Predictions RNN")
            print(predictions)

        caption_loss = loss_function(target, predictions)  # (batch_size, )

        # After tape
        total_batch_loss = tf.reduce_sum(caption_loss)  # Sum (batch_size, ) => K
        mean_batch_loss = tf.reduce_mean(caption_loss)  # Mean(batch_size, ) => K

    # Updated the variables
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(caption_loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return total_batch_loss, mean_batch_loss
# Start Training
loss_plot = []
##################
EPOCHS = 50
start_epoch = 0

for epoch in range(start_epoch, EPOCHS):
    real_epoch = len(loss_plot) + 1
    start = time()
    total_loss = 0

    for (batch, (img_tensor, target)) in enumerate(dataset):
        total_batch_loss, mean_batch_loss = train_step(img_tensor, target, verbose=False)
        total_loss += total_batch_loss

        if batch % 100 == 0:
            print('Epoch {} Batch {} Batch Loss {:.4f}'.format(real_epoch, batch, mean_batch_loss.numpy()))

    print('Total Loss {:.6f}'.format(total_loss))
    epoch_loss = total_loss / NUM_STEPS

    # storing the epoch end loss value to plot later
    loss_plot.append(epoch_loss)

    if epoch % 5 == 0:
        ckpt_manager.save()

    print('Epoch {} Epoch Loss {:.6f}'.format(real_epoch, epoch_loss))
    print('Time taken for 1 epoch {} sec\n'.format(time() - start))
##################
plt.plot(loss_plot)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Loss Plot')
plt.show()
################### /// Test Stage /// ####################
# Evaluate random image
# Remove <start>, <end> and <pad> marks from the predicted sequence
def clean_caption(caption):
    return [item for item in caption if item not in ['<start>', '<end>', '<pad>']]

test_img_name = random.choice(image_filenames_train)
##################
######### ///// LSTM ///// #################
##################
# Get captions from a test image
def get_caption(img):
    # Add image to an array to simulate batch size of 1
    features = encoder(tf.expand_dims(img, 0))

    caption = []
    dec_input = tf.expand_dims([], 0)

    # Inputs the image embedding into the trained LSTM layer and predicts the first word of the sequence.
    # The output, hidden and cell states are passed again to the LSTM to generate the next word.
    # The iteration is repeated until the caption does not reach the max length.
    state = None
    for i in range(1, max_caption_length):
        predictions, memory_state, carry_state = \
            decoder(dec_input, features, omit_features=i > 1, initial_state=state)

        # Takes maximum index of predictions
        word_index = np.argmax(predictions.numpy().flatten())

        caption.append(tokenizer.index_word[word_index])

        dec_input = tf.expand_dims([word_index], 0)
        state = [memory_state, carry_state]

    # Filter caption
    return clean_caption(caption)

raw_img = load_image(test_img_name)[0]
img = images_dict[test_img_name]
captions = images_captions_dict[test_img_name]

plt.imshow(raw_img)

print("Real captions")
for caption in captions:
    print(caption)

print("Esimated caption")
estimated_caption = get_caption(img)
print(estimated_caption)

# # Evaluate dataset using BLEU
def get_caption(img):
    # Add image to an array to simulate batch size of 1
    features = encoder(tf.expand_dims(img, 0))

    caption = []
    dec_input = tf.expand_dims([], 0)

    state = None
    for i in range(1, max_caption_length):
        predictions, memory_state, carry_state = \
            decoder(dec_input, features, omit_features=i > 1, initial_state=state)

        word_index = np.argmax(predictions.numpy().flatten())

        caption.append(tokenizer.index_word[word_index])

        dec_input = tf.expand_dims([word_index], 0)
        state = [memory_state, carry_state]

    # Filter caption
    return clean_caption(caption)

actual, predicted = [], []

for test_img_name in image_filenames_test:
    img = images_dict[test_img_name]
    estimated_caption = get_caption(img)

    captions = [clean_caption(caption.split()) for caption in images_captions_dict[test_img_name]]

    # store actual and predicted
    actual.append(captions)
    predicted.append(estimated_caption)

# Print BLEU score
print('BLEU-1: %f' % corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0)))
print('BLEU-2: %f' % corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0)))

#decoder.save('./saved_models/decoder-LSTM')
