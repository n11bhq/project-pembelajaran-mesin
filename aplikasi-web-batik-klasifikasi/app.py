import time
import os
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, redirect, render_template
from tensorflow import keras
#from tensorflow.keras.models import load_model
#from tensorflow.keras.models import model_from_json

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app = Flask(__name__)
UPLOAD_FOLDER = 'static/uploads/'
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

@app.after_request
def add_header(r):
    r.headers["Cache-Control"] = "no-cache, no-store, must-revalidate"
    r.headers["Pragma"] = "no-cache"
    r.headers["Expires"] = "0"
    r.headers['Cache-Control'] = 'public, max-age=0'
    return r

@app.route("/")
def index():
    return render_template('/select.html', )

@app.route('/predict', methods=['POST'])
def predict():
    chosen_model = request.form['select_model']
    model_dict = {'CNN'   : "F:\\UMM\\ML\\Praktikum\\project-pembelajaran-mesin\\aplikasi-web-batik-klasifikasi\\model\\batik_model_2.h5",
                  'Transferlearning'     :   "F:\\UMM\\ML\\Praktikum\\project-pembelajaran-mesin\\aplikasi-web-batik-klasifikasi\\model\\batik_tlresnet_v2.h5",}
    if chosen_model in model_dict:
        model = keras.models.load_model(model_dict[chosen_model]) 
    else:
        model = keras.models.load_model(model_dict[0])
    file = request.files["file"]
    file.save(os.path.join('F:\\UMM\ML\\Praktikum\\project-pembelajaran-mesin\\aplikasi-web-batik-klasifikasi\\static', 'temp.jpg'))
    img = cv2.cvtColor(np.array(Image.open(file)), cv2.COLOR_BGR2RGB)
    img = np.expand_dims(cv2.resize(img, model.layers[0].input_shape[0][1:3] if not model.layers[0].input_shape[1:3] else model.layers[0].input_shape[1:3]).astype('float32') / 255, axis=0)
    start = time.time()
    pred = model.predict(img)[0]
    predicted_class_indices = np.argmax(pred)
    value = "empty"
    if predicted_class_indices == 0:
        value = 'Ceplok'
    elif predicted_class_indices == 1:
        value = 'Kawung'
    elif predicted_class_indices == 2:
        value = 'Lereng'
    elif predicted_class_indices == 3:
        value = 'Mix_motif'
    elif predicted_class_indices == 4:
        value = 'Nitik'
    else:
        value = 'Parang'
    runtimes = round(time.time()-start,4)
    respon_model = [round(elem * 100, 2) for elem in pred]
    return predict_result(value, runtimes, respon_model, 'temp.jpg')

def predict_result(model, run_time, probs, img):
    class_list = {'Ceplok': 0, 'Kawung': 1, 'Lereng': 2, 'Mix_motif': 3, 'Nitik': 4, 'Parang': 5}
    idx_pred = probs.index(max(probs))
    labels = list(class_list.keys())
    return render_template('/result_select.html', labels=labels, 
                            probs=probs, model=model, pred=idx_pred, 
                            run_time=run_time, img=img)

if __name__ == "__main__": 
    app.run(host="127.0.0.1", port=8080, debug=True)
