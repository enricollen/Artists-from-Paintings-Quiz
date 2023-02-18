from flask import Flask, render_template, request, redirect, url_for
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress tensorflow warnings
# from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing import image
from keras import models
import numpy as np
import random

app = Flask(__name__)

CLASSES = ['Paul_Gauguin',
           'Edgar_Degas',
           'Vincent_van_Gogh',
           'Albrecht_Durer',
           'Pablo_Picasso',
           'Titian',
           'Francisco_Goya',
           'Marc_Chagall',
           'Pierre-Auguste_Renoir',
           'Alfred_Sisley',
           'Rembrandt']

CLASSES.sort()

# Define a list of images and their corresponding labels
images = []

# to not choose 2 images from same artist
image_prec = []

# first iteration
first_iter = True

# Define a counter to keep track of how many images have been shown
counter = 0

# Define the current score
score = 0

# Set the path to the directory containing the subfolders
dir_path = "static/paintings"

# Initialize an empty list to store the selected images
selected_images = []

# Initialize an empty array to store the directory names
dir_names = []

# Initialize the predictor model
model = None

# AI player score
AI_score = 0


def pick_random_paintings():
    global images, selected_images, dir_path, dir_names

    # Get a list of all subfolders in the directory
    subfolders = [f.path for f in os.scandir(dir_path) if f.is_dir()]

    # Randomly select 4 distinct subfolders for 4 different buttons
    selected_subfolders = random.sample(subfolders, 4)

    # Loop through each selected subfolder and randomly select an image
    for subfolder in selected_subfolders:
        # Get a list of all image files in the subfolder
        images = [f.path for f in os.scandir(subfolder) if f.is_file() and f.name.endswith((".jpg", ".jpeg", ".png"))]
        # Randomly select 5 image for each author selected
        for img in range(5):
            selected_image = random.choice(images)
            # Add the selected image to the list of selected images
            selected_images.append(selected_image)

    # Loop through each file path in the array
    for file_path in selected_images:
        # Get the parent directory name from the file path
        dir_name = os.path.basename(os.path.dirname(file_path))
        # Add the directory name to the list of directory names
        dir_names.append(dir_name)


# Define a function to get a random image from the list
def get_image():
    global counter, images

    if counter >= 10:
        return None

    return random.choice(images)


# Define the home page route
@app.route('/')
def index():
    global first_iter, image_prec, counter, AI_score

    # Get a random image from the list
    if first_iter == True:
        image = get_image()
        image_prec.append(image)
        first_iter = False
    else:
        image = get_image()
        while image in image_prec:
            image = get_image()
        image_prec.append(image)

    # increment user trials
    counter += 1

    # make prediction for AI Player and update its score
    if image is not None:  # checks whether is the last iteration
        predicted_artist = compute_classifier_predictions(image[0])  # in 0 there's the img url
        actual_artist = image[1]
        if predicted_artist == actual_artist:
            AI_score += 1

    if image is None:
        # Reset the counter
        global score
        counter = 0
        # Render the final score template
        return render_template('result.html', score=score, scoreAI=AI_score)
    # Shuffle the labels and select the first four
    labels = [image[1]]
    while len(labels) < 4:
        label = random.choice([img[1] for img in images])
        if label not in labels:
            labels.append(label)
    random.shuffle(labels)
    # Render the home page template with the image URL and the shuffled labels
    return render_template('index.html', image_url=image[0], labels=labels)


# Define the guess route
@app.route('/guess', methods=['POST'])
def guess():
    # Get the user's guess from the form data
    guess = request.form['guess']
    # Get the correct label for the image
    correct_label = request.form['correct_label']
    # Check if the guess is correct and update the score
    global score
    if guess == correct_label:
        score += 1

    # Redirect to the home page to show the next image
    return redirect(url_for('index'))


# Define the restart route
@app.route('/restart')
def restart():
    global counter, score, images, selected_images, first_iter, image_prec, AI_score, dir_names

    # reset the game variables
    reset_variables()
    # pick again random imgs that updates images and selected_images again
    pick_random_paintings()
    images = list(zip(selected_images, dir_names))

    # Redirect to the home page to start a new game
    return redirect(url_for('index'))


def reset_variables():
    # Reset all the global variables
    global counter, score, images, selected_images, first_iter, image_prec, AI_score, dir_names

    counter = 0
    score = 0
    images = []
    selected_images = []
    first_iter = True
    image_prec = []
    dir_names = []
    AI_score = 0


def load_classifier():
    global model
    model = models.load_model('./models/VGG_NoDropout.h5', compile=False)


def compute_classifier_predictions(selected_image_path):
    global CLASSES

    img = image.load_img(selected_image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_batch = np.expand_dims(img_array, axis=0)
    img_preprocessed = preprocess_input(img_batch)
    index_label = int(np.argmax(model.predict(img_preprocessed), axis=-1))
    label_name = CLASSES[index_label]
    return label_name


if __name__ == '__main__':
    load_classifier()
    pick_random_paintings()
    images = list(zip(selected_images, dir_names))

    app.run(debug=False)
