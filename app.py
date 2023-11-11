from ultralytics import YOLO
from flask import Flask, render_template, request, flash, url_for, redirect
import os

from transformers import ViTImageProcessor, ViTForImageClassification # ViTFeatureExtractor
from PIL import Image

import warnings
warnings.simplefilter('ignore')

app = Flask(__name__)

# Creates a Class to Store the Filename between Pages
class DataStore():
    filename = None
name=DataStore()

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = "super_secret_key"  # for flashing messages

# Ensure the upload directory exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route("/", methods=["GET","POST"]) # Defines the Home Page
def home():
    if request.method == "POST":  # Defines the return Home button
        return render_template("home.html") # Returns to the Home Page

    return render_template("home.html") # Returns to the Home Page

@app.route('/upload_ViT', methods=['POST']) # Defines the Upload ViT Page
def upload_file_ViT():

    # check if the post request has the file part
    if 'file' not in request.files:
        return render_template("upload_ViT.html")
    
    file = request.files['file']
    
    # if user does not select file, browser also submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return render_template("upload_ViT.html")
        
    # if user does selects a file, upload it and move to Predict_ViT Webpage
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        name.filename=filename
        file.save(filename)
        return redirect(url_for('predict_ViT'))
    
    flash('Invalid file type. Please upload an image.')
    return render_template("upload_ViT.html")
    

@app.route('/predict_ViT', methods=['GET','POST']) # Defines the Predict ViT Page
def predict_ViT():

    # Load image with PIL module
    img = name.filename
    image = Image.open(img)

    feature_extractor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224') # Instantiate the feature extractor specific to the model checkpoint
    model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224') # Instantiate the pretrained model
    inputs = feature_extractor(images=image, return_tensors="pt") # Extract features (patches) from the image
    outputs = model(**inputs) # Predict by feeding the model (** is a python operator which unpacks the inputs)
    logits = outputs.logits # Convert outputs to logis
    predicted_class_idx = logits.argmax(-1).item() # model predicts one of the classes by pick the logit which has the highest probability
    predication_ViT = model.config.id2label[predicted_class_idx] # Gives the Predication and Stores it

    return render_template("predict_ViT.html", predication_ViT=predication_ViT) # Sends predication_ViT to the predict ViT webpage

@app.route('/upload_Yolo', methods=['POST']) # Defines the Upload Yolo Page
def upload_file_Yolo():

    # check if the post request has the file part
    if 'file' not in request.files:
        return render_template("upload_Yolo.html")
    
    file = request.files['file']
    
    # if user does not select file, browser also submit an empty part without filename
    if file.filename == '':
        flash('No selected file')
        return render_template("upload_Yolo.html")
    
    # if user does selects a file, upload it and move to Predict_Yolo Webpage
    if file and allowed_file(file.filename):
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        name.filename=filename
        file.save(filename)
        flash('File successfully uploaded and saved!')
        return redirect(url_for('predict_Yolo'))
    
    flash('Invalid file type. Please upload an image.')
    return render_template("upload_Yolo.html")

@app.route('/predict_Yolo', methods=['GET','POST']) # Defines the Predict Yolo Page
def predict_Yolo():

    model = YOLO('yolov8n.pt') # Load a pretrained YOLOv8n model
    results = model(str(name.filename), verbose=False)  # results list 

    # Save results as image
    for r in results:
        im_array = r.plot()  # plot a BGR numpy array of predictions
        im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
        #im.show()  # show image
        im.save('static/media/results.jpg')  # save image

    return render_template("predict_Yolo.html")

app.run(debug=False, port = 5000)
