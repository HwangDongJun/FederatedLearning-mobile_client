import flask
import werkzeug
from flask import Flask
from flask import request
from flask import send_file
from flask import send_from_directory
from flask import render_template
from flask_socketio import SocketIO

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import numpy as np
import http.server
import threading
import os
import json
import pickle
import time

from model_evaluate import evaluate_LocalModel
 
app = Flask(__name__)
socketio = SocketIO(app)

#ip_address = '192.168.1.124'
#ip_address = '192.168.0.100'
ip_address = '192.168.0.8'

MODEL_VERSION = 0
MIN_NUM_WORKERS = 1
MAX_NUM_ROUNDS = 4
NUM_CLIENTS_CONTACTED_PER_ROUND = 0
#currentRound = 0
ALL_CURRENT_ROUND = 0
currentRoundClientUpdates = list()
readyClientSids = list()
made_model = None
clientUpdateAmount = 0
aggregation_username = list()

global_model_loss = 0.0
global_model_acc = 0.0

HEIGHT = 224
WIDTH = 224
channels = 3
lr = 0.001

@app.route('/')
def main_test():
    return "main test"

@app.route('/client_wake_up')
def wake_up_client():
    rec_client_name = request.args.get('client_name', "user")
    readyClientSids.append(rec_client_name)

    print(readyClientSids)
    print('init model')
    time.sleep(15)
    return "success client check"

@app.route('/client_ready')
def ready_client():
    global ALL_CURRENT_ROUND; global NUM_CLIENTS_CONTACTED_PER_ROUND

    ready_client = request.args.get('client_name', "user")
    train_size = request.args.get('train_size', '0')
    model_ver = request.args.get('model_version', "0")
    curr_round = request.args.get('current_round', "0")
    print("client ready : " + ready_client + " train size : " + train_size + " model version : " + model_ver)

    MODEL_VERSION = model_ver
    aggregation_username.append(ready_client)
    NUM_CLIENTS_CONTACTED_PER_ROUND += 1

    #if ready_client in readyClientSids and len(readyClientSids) >= MIN_NUM_WORKERS and currentRound == 0:
    if ready_client in readyClientSids and int(curr_round) == 0:
        print("### Check Train Round ###")
        model_loss, model_acc = trainNextRound(curr_round)

        ALL_CURRENT_ROUND = int(curr_round) + 1
        wrap_json_info = json.dumps({
            'model_version': model_ver,
            'current_round': ALL_CURRENT_ROUND,
            'max_train_round': MAX_NUM_ROUNDS,
            'model_loss': model_loss,
            'model_acc': model_acc,
            'upload_url': "http://" + ip_address + ":8891/upload",
            'model_url': "http://" + ip_address + ":8891/download"})

        print(wrap_json_info)

        return wrap_json_info
    else:
        return "NOT REGISTER CLIENT"

@app.route('/client_update')
def update_client():
    global clientUpdateAmount; global currentRoundClientUPdates
    global ALL_CURRENT_ROUND
    global global_model_loss; global global_model_acc

    update_client = request.args.get('client_name', "user")
    current_round = request.args.get('current_round', "0")

    if int(ALL_CURRENT_ROUND) == int(current_round):
        clientUpdateAmount += 1
        currentRoundClientUpdates.append(update_client)
        print("### client update weights ###")

        if clientUpdateAmount >= NUM_CLIENTS_CONTACTED_PER_ROUND and len(currentRoundClientUpdates) > 0:
            print("### Received All Client ###")
            updateWeight()
            print("### update global model success ###")
            if int(current_round) >= int(MAX_NUM_ROUNDS):
                print("### finish all training!!! ###")
                eval_res = stopAndEval()

                ALL_CURRENT_ROUND = int(current_round) + 1
                wrap_json_info_eval = json.dumps({
                    'model_version': MODEL_VERSION,
                    'current_round': ALL_CURRENT_ROUND,
                    'model_loss': eval_res['loss'],
                    'model_acc': eval_res['accuracy'],
                    'state': 'FIN'})
                return wrap_json_info_eval
            else:
                clientUpdateAmount -= 1

                model_loss, model_acc = trainNextRound(current_round)
                global_model_loss = model_loss
                global_model_acc = model_acc

                ALL_CURRENT_ROUND = int(current_round) + 1
                wrap_json_info = json.dumps({
                    'model_version': MODEL_VERSION,
                    'current_round': ALL_CURRENT_ROUND,
                    'model_loss': model_loss,
                    'model_acc': model_acc,
                    'upload_url': "http://" + ip_address + ":8891/upload",
                    'model_url': "http://" + ip_address + ":8891/download",
                    "state" : "RESP_ARY"})
                return wrap_json_info # "No train from all rounds yet."
        
        wrap_json_info = json.dumps({
            'model_version': MODEL_VERSION,
            'current_round': ALL_CURRENT_ROUND,
            'state': "RESP_ACY"})

        return wrap_json_info # "No response from all clients yet."

    wrap_json_info = json.dumps({
        'model_version': MODEL_VERSION,
        'state': "NEC"})

    return wrap_json_info # "Not equal currentRound"

@app.route('/download_html')
def upload_form():
    return render_template('download.html')

@app.route('/download')
def download_file():
    return send_from_directory(directory='/home/dnlab2020/federated_socketio_server/saved_model', filename='MyMultiLayerNetwork.zip', as_attachment=True)
    #return send_file("/home/dnlab2020/federated_socketio_server/saved_model/MyMultiLayerNetwork.zip", as_attachment=True)

@app.route('/upload', methods=['GET', 'POST'])
def upload_model_file():
    modelfile = flask.request.files['files']
    filename = werkzeug.utils.secure_filename(modelfile.filename)
    print("\nReceived Model File name : " + modelfile.filename)
    modelfile.save(filename)
    return "file_success"

@app.route('/model_version')
def check_model_version():
    global clientUpdateAmount

    version_client = request.args.get('version_client', "user0")
    print("Request Check Model Version " + version_client)
    model_ver = request.args.get('model_ver', "0")

    if int(model_ver) == int(MODEL_VERSION):
        return json.dumps({'state': "WAIT"})
    else:
        clientUpdateAmount -= 1

        if ALL_CURRENT_ROUND >= MAX_NUM_ROUNDS:
            wrap_json_info = json.dumps({
                'model_version': MODEL_VERSION,
                'current_round': ALL_CURRENT_ROUND,
                'model_loss': global_model_loss,
                'model_acc': global_model_acc,
                'upload_url': "http://" + ip_address + ":8891/upload",
                'model_url': "http://" + ip_address + ":8891/download",
                'state': "FIN"}) # FINISH ALL TRAINING ROUNDS
        else:
            wrap_json_info = json.dumps({
                'model_version': MODEL_VERSION,
                'current_round': ALL_CURRENT_ROUND,
                'model_loss': global_model_loss,
                'model_acc': global_model_acc,
                'upload_url': "http://" + ip_address + ":8891/upload",
                'model_url': "http://" + ip_address + ":8891/download",
                'state': "NW"}) # NOT WAIT

        return wrap_json_info

def stopAndEval():
    print("### Finish All Round ###")

    # eval
    model_evalResult = eval_model.train_model_tosave()
    print("Evaluate Loss : " + str(model_evalResult['loss']) + " Evaluate Accuracy : " + str(model_evalResult['accuracy']))

    # 이후에 각 클라이언트로 결과를 전송하기 위해서는 socket?구현이 필요한가?
    return model_evalResult

def updateWeight():
    global MODEL_VERSION

    model_weight = made_model.get_weights()

    set_model_weight = list()
    for ag in aggregation_username:
        os.system(f'mv ./weight_{ag}.json client_model')
        with open(f'./client_model/weight_{ag}.json', 'r') as json_fr:
            weight_json = json.load(json_fr)

        json_keys = list(weight_json.keys())
        new_model_weight = list()
        index = 0; weight_count = 0
        while len(json_keys) != 0:
            if str(index) + "_W" in json_keys:
                weight_shape_w = model_weight[weight_count].shape
                new_model_weight.append(np.array(weight_json[str(index) + "_W"]).reshape(weight_shape_w))
                json_keys.remove(str(index) + "_W")
                weight_count += 1
            elif str(index) + "_b" in json_keys:
                weight_shape_b = model_weight[weight_count].shape
                new_model_weight.append(np.array(weight_json[str(index) + "_b"]).reshape(weight_shape_b))
                json_keys.remove(str(index) + "_b")
                index += 1
                weight_count += 1
            else:
                index += 1

        if len(set_model_weight) == 0:
            set_model_weight = new_model_weight
        else:
            for i, wc in enumerate(new_model_weight):
                set_model_weight[i] += wc

    made_model.set_weights(new_model_weight)
    MODEL_VERSION += 1 # upgrade model version

def trainNextRound(currentRound):
    global currentRoundClientUpdates

    #currentRound += 1
    currentRoundClientUpdates.clear()
    print("### Round " + str(currentRound) + " ###")

    # eval
    model_evalResult = eval_model.train_model_tosave()
    print("Evaluate Loss : " + str(model_evalResult['loss']) + " Evaluate Accuracy : " + str(model_evalResult['accuracy']))

    try:
        # save
        print("### Model save (config, weight) ###")
        save_model()
        # convert config, weight to zip
        print("### Model convert to zip file ###")
        convert2model()
    except OSError as oe:
        print(oe)
        time.sleep(10)
        save_model()
        convert2model()
    
    return model_evalResult['loss'], model_evalResult['accuracy']

def save_model():
    model_json = made_model.to_json()
    with open("./saved_model/model_config.json", "w") as f:
        f.write(model_json)

    made_model.save_weights('./saved_model/model_weights.h5')

def convert2model():
    os.system("cd model_convert && mvn exec:java -D exec.mainClass=org.example.convert2model")

def buildGlobalModel():
    #feature_vector_url = "https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4"
    #feature_vector_layer = hub.KerasLayer(feature_vector_url, input_shape=(HEIGHT, WIDTH, channels))
    #feature_vector_layer.trainable = True

    #model = tf.keras.Sequential([
    #    feature_vector_layer,
    #    layers.Dense(5, activation='softmax')
    #])
    #print(model.summary())
    #model.compile(
    #    optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
    #    loss='categorical_crossentropy',
    #    metrics=['categorical_accuracy'])

    #mobilenet_v2 = tf.keras.applications.MobileNetV2(
    #    weights='imagenet',
    #    input_shape=(HEIGHT, WIDTH, channels),
    #    include_top=False,
    #    pooling='max')

    #model = tf.keras.Sequential([
    #    mobilenet_v2,
    #    layers.Dense(5, activation='softmax')
    #])

    # Functional
    #inputs = layers.Input(shape=(HEIGHT, WIDTH, channels))
    #x = inputs
    #x = mobilenet_v2(x)
    #x = layers.Dense(5, activation='softmax')(x)
    #model = keras.Model(inputs=inputs, outputs=x)

    model = tf.keras.Sequential([
        layers.InputLayer(input_shape=(HEIGHT, WIDTH, channels)),
        layers.Conv2D(10, (5, 5), strides=(1, 1), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(10, (5, 5), strides=(1, 1), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(10, (5, 5), strides=(1, 1), activation='relu'),
        layers.Conv2D(10, (5, 5), strides=(1, 1), activation='relu'),
        layers.Flatten(),
        layers.Dense(10, activation='relu'),
        layers.Dense(5, activation='softmax')
    ])
#    inputs = layers.Input(shape=(channels, HEIGHT, WIDTH))
#    x = layers.Conv2D(20, (5, 5), padding='same', activation='relu')(inputs)
#    x = layers.MaxPooling2D((2, 2))(x)
#    x = layers.Conv2D(40, (5, 5), padding='same', activation='relu')(x)
#    x = layers.MaxPooling2D((2, 2))(x)
#    x = layers.Conv2D(30, (5, 5), padding='same', activation='relu')(x)
#    x = layers.Conv2D(10, (5, 5), padding='same', activation='relu')(x)
#    x = layers.Flatten()(x)
#    x = layers.Dense(10, activation='relu')(x)
#    x = layers.Dense(5, activation='softmax')(x)
#    model = keras.Model(inputs=inputs, outputs=x)

    print(model.summary())
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='categorical_crossentropy',
        metrics=['accuracy'])

    #model.save('./saved_model/init_model.h5')
    #model_json = model.to_json()
    #with open('./saved_model/model_config.json', 'w') as fw:
    #    fw.write(model_json)

    #model.save_weights('./saved_model/model_weights.h5')

    return model

if __name__ == "__main__":
    made_model = buildGlobalModel()
    eval_model = evaluate_LocalModel(made_model, 16, HEIGHT, np.array(['book', 'laptop', 'phone', 'wash', 'water']))
    app.run(host="0.0.0.0", port=8891, debug=True, threaded=True)
