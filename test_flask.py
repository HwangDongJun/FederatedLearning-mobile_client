import flask
import werkzeug
from flask import Flask
from flask import request
from flask import send_file
from flask import send_from_directory
from flask import render_template, redirect, url_for
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
import sqlite3
import subprocess
import netifaces as ni
import random

from model_evaluate import evaluate_LocalModel

con = sqlite3.connect("/home/dnlab2020/federated_socketio_server/db_file/fl_dashboard.db", check_same_thread=False)

app = Flask(__name__)
socketio = SocketIO(app)

#ip_address = '192.168.1.124'
#ip_address = '192.168.0.108'
ip_address = '192.168.0.8'

CHECK_CLIENT_TRAINING = False
CHECK_TRAIN_TIMER = False
CHECK_MODEL_UPDATE = False
MODEL_VERSION = 0
MODEL_TRAINING_TIME = 0
MIN_NUM_WORKERS = 1
MAX_NUM_ROUNDS = 20
NUM_CLIENTS_CONTACTED_PER_ROUND = 0
#currentRound = 0
ALL_CURRENT_ROUND = 0

currentRoundClientUpdates = list()
readyClientSids = list()
made_model = None
clientUpdateAmount = 0
aggregation_username = list()
waiting_username = list()
training_username = list()
participate_username = list()
wake_up_client_count = 0

global_model_loss = 0.0
global_model_acc = 0.0

HEIGHT = 112
WIDTH = 112
channels = 3
lr = 0.001

# timer
def client_training_check_timer():
	global CHECK_CLIENT_TRAINING

	print("Start timer waiting for client's request.")
	time.sleep(10)
	print("Waiting for the requesting client, learning the requested client.")
	CHECK_CLIENT_TRAINING = True

def next_round_train_timer():
	time.sleep(60)

def wifi_info():
	output = subprocess.check_output(['sudo', 'iwgetid'])
	interface_name = output.split()[0].decode()
	ssid_name = output.split()[1].decode().split('"')[1]
	ip_addr = ni.ifaddresses(interface_name)[ni.AF_INET][0]['addr']

	return ssid_name, ip_addr

@app.route('/dashboard', methods=['POST', 'GET'])
def main_page():
	client_info = dict()
	cur = con.cursor()
	cur.execute('''SELECT ClientInfo.username,ClientInfo.androidname,ClientInfo.trainsize,ClientInfo.classsize,DeviceInfo.battery_pct,DeviceInfo.is_charging 
					FROM ClientInfo INNER JOIN DeviceInfo
					ON ClientInfo.username = DeviceInfo.username''')

	print("train done", waiting_username)
	print("train", training_username)
	print("participate", participate_username)
	print("on, off", readyClientSids)

	# wifi
	Nssid, Aip = wifi_info()

	count_list = [0, 0, 0, 0, 0]

	for row in cur:
		if row[0] in participate_username:
			client_info[row[0]] = [row[1], row[2], row[3], "participate", float(row[4]), row[5]]
			count_list[0] += 1
		elif row[0] in waiting_username:
			client_info[row[0]] = [row[1], row[2], row[3], "wait", float(row[4]), row[5]]
			count_list[1] += 1
		elif row[0] in training_username:
			client_info[row[0]] = [row[1], row[2], row[3], "train", float(row[4]), row[5]]
			count_list[2] += 1
		elif row[0] in readyClientSids:
			client_info[row[0]] = [row[1], row[2], row[3], "on", float(row[4]), row[5]]
			count_list[3] += 1
		elif row[0] not in readyClientSids:
			client_info[row[0]] = [row[1], row[2], row[3], "off", float(row[4]), row[5]]
			count_list[4] += 1

	return render_template('MainPage.htm', ClientInfo=client_info, CountList=count_list, WifiInfo={Nssid:Aip})

@app.route('/traininfo', methods=['POST', 'GET'])
def traininfo_page():
	# TrainInfo Page Information
	CLIENT_LABELS = list(); ROUND_LABELS = list()
	CLIENT_MODEL_ACCURACY = list()
	CLIENT_MODEL_F1 = list()
	CLIENT_MODEL_TRAININGTIME = list()
	CLIENT_MODEL_CLASS_DATA = list()

	train_current_status = dict(); colors = list()
	cur = con.cursor()
	cur.execute('''SELECT * FROM TrainInfo''')

	# wifi
	Nssid, Aip = wifi_info()

	for row in cur:
		if row[0] in readyClientSids:
			train_current_status[row[0]] = [row[3], row[4]]
			# round labels
			ROUND_LABELS = list(range(row[3]))[1:]

			data_acc_dict = dict(); data_f1_dict = dict()
			data_trainingtime_dict = dict()
			# 1. client labels
			data_acc_dict["label"] = row[0]; data_f1_dict["label"] = row[0]
			data_trainingtime_dict["label"] = row[0]
			# 2. color - acc
			one_color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
			data_acc_dict["backgroundColor"] = one_color
			data_acc_dict["borderColor"] = one_color
			data_acc_dict["fill"] = False
			# n. accuracy
			data_acc_dict["data"] = list(map(float, row[1].split(',')))
			if len(CLIENT_MODEL_ACCURACY) != 0:
				for index, cma in enumerate(CLIENT_MODEL_ACCURACY):
					if cma['label'] == row[0]:
						CLIENT_MODEL_ACCURACY[index] = data_acc_dict
					else:
						CLIENT_MODEL_ACCURACY.append(data_acc_dict)
			else:
				CLIENT_MODEL_ACCURACY.append(data_acc_dict)
			# 2. color - loss
			data_loss_dict["backgroundColor"] = one_color
			data_loss_dict["borderColor"] = one_color
			data_loss_dict["fill"] = False
			# n. loss
			data_f1_dict["data"] = list(map(float, row[2].split(',')))
			if len(CLIENT_MODEL_F1) != 0:
				for index, cml in enumerate(CLIENT_MODEL_F1):
					if cml['label'] == row[0]:
						CLIENT_MODEL_F1[index] = data_f1_dict
					else:
						CLIENT_MODEL_F1.append(data_f1_dict)
			else:
				CLIENT_MODEL_F1.append(data_f1_dict)

			# 2. color - training time
			data_trainingtime_dict["backgroundColor"] = one_color
			data_trainingtime_dict["borderColor"] = one_color
			data_trainingtime_dict["fill"] = False
			# n. training time
			data_trainingtime_dict["data"] = list(map(float, row[5].split(',')))
			if len(CLIENT_MODEL_TRAININGTIME) != 0:
				for index, cmt in enumerate(CLIENT_MODEL_TRAININGTIME):
					if cmt['label'] == row[0]:
						CLIENT_MODEL_TRAININGTIME[index] = data_trainingtime_dict
					else:
						CLIENT_MODEL_TRAININGTIME.append(data_trainingtime_dict)
			else:
				CLIENT_MODEL_TRAININGTIME.append(data_trainingtime_dict)

	print("round label", ROUND_LABELS)
	print("accuracy", CLIENT_MODEL_ACCURACY)
	print("f1 score", CLIENT_MODEL_F1)

	cur.execute('''SELECT * FROM ClassInfo''')
	class_labels = list(); data_size_list = list(); class_backcolor = list()
	for row in cur:
		class_count = row[1]
		class_data_list = row[2].split('/')[:-1]
		for cdl in class_data_list:
			class_label = cdl.split('-')[0]
			class_length = cdl.split('-')[1]
			if class_label not in class_labels:
				class_labels.append(class_label)
				data_size_list.append(int(class_length))
			else:
				data_size_list[class_labels.index(class_label)] += int(class_length)
	
	client_model_class_data_dict = dict(); sub_dataset_dict = dict()
	# 1. label
	client_model_class_data_dict['labels'] = class_labels
	# 2. datasets
		# 2-1. data
	sub_dataset_dict['data'] = data_size_list
		# 2-2. color
	for i in range(len(class_labels)):
		class_backcolor.append("#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)]))
	sub_dataset_dict['backgroundColor'] = class_backcolor
	sub_dataset_dict['borderWidth'] = 0
	sub_dataset_dict['label'] = "Client Dataset"
	client_model_class_data_dict['datasets'] = [sub_dataset_dict]

	CLIENT_MODEL_CLASS_DATA.append(client_model_class_data_dict)

	print("client class", CLIENT_MODEL_CLASS_DATA)

	return render_template('TrainPage.htm', TrainCurrentInfo=train_current_status, RoundLabel=ROUND_LABELS, TrainModelAcc=CLIENT_MODEL_ACCURACY, TrainModelF1=CLIENT_MODEL_F1, TrainModelTrainingTime=CLIENT_MODEL_TRAININGTIME, ClientClass=CLIENT_MODEL_CLASS_DATA, WifiInfo={Nssid:Aip})
	
@app.route('/deviceinfo', methods=['POST', 'GET'])
def deviceinfo_page():
	# TrainInfo Page Information
	CLIENT_LABELS = list(); ROUND_LABELS = list()
	CLIENT_MODEL_HEAPSIZE = list()
	CLIENT_MODEL_TEMPERATURE = list()
	CLIENT_MODEL_CPU_FREQ = list()

	train_current_status = dict(); colors = list()
	cur = con.cursor()
	cur.execute('''SELECT * FROM TrainInfo''')

	# wifi
	Nssid, Aip = wifi_info()

	for row in cur:
		if row[0] in readyClientSids:
			train_current_status[row[0]] = [row[3], row[4]]
			# round labels
			ROUND_LABELS = list(range(row[3]))[1:]

			data_heapsize_dict = dict()
			data_temperature_dict = dict(); data_cpu_freq_dict = dict()
			# 1. client labels
			data_heapsize_dict["label"] = row[0]
			data_temperature_dict["label"] = row[0]; data_cpu_freq_dict["label"] = row[0]

			# 2. color - heap size
			one_color = "#"+''.join([random.choice('0123456789ABCDEF') for j in range(6)])
			data_heapsize_dict["backgroundColor"] = one_color
			data_heapsize_dict["borderColor"] = one_color
			data_heapsize_dict["fill"] = False
			# n. heap size
			data_heapsize_dict["data"] = list(map(int, row[6].split(',')))
			if len(CLIENT_MODEL_HEAPSIZE) != 0:
				for index, cmh in enumerate(CLIENT_MODEL_HEAPSIZE):
					if cmh['label'] == row[0]:
						CLIENT_MODEL_HEAPSIZE[index] = data_heapsize_dict
					else:
						CLIENT_MODEL_HEAPSIZE.append(data_heapsize_dict)
			else:
				CLIENT_MODEL_HEAPSIZE.append(data_heapsize_dict)

			# 2. color - cpu frequency
			data_cpu_freq_dict["backgroundColor"] = one_color
			data_cpu_freq_dict["borderColor"] = one_color
			data_cpu_freq_dict["fill"] = False
			# n. cpu frequency
			data_cpu_freq_dict["data"] = list(map(float, row[7].split(',')))
			if len(CLIENT_MODEL_CPU_FREQ) != 0:
				for index, cmcf in enumerate(CLIENT_MODEL_CPU_FREQ):
					if cmcf['label'] == row[0]:
						CLIENT_MODEL_CPU_FREQ[index] = data_cpu_freq_dict
					else:
						CLIENT_MODEL_CPU_FREQ.append(data_cpu_freq_dict)
			else:
				CLIENT_MODEL_CPU_FREQ.append(data_cpu_freq_dict)

			# 2. color - temperature
			data_temperature_dict["backgroundColor"] = one_color
			data_temperature_dict["borderColor"] = one_color
			data_temperature_dict["fill"] = False
			# n. temperature
			data_temperature_dict["data"] = list(map(float, row[8].split(',')))
			if len(CLIENT_MODEL_TEMPERATURE) != 0:
				for index, cmt in enumerate(CLIENT_MODEL_TEMPERATURE):
					if cmt['label'] == row[0]:
						CLIENT_MODEL_TEMPERATURE[index] = data_temperature_dict
					else:
						CLIENT_MODEL_TEMPERATURE.append(data_temperature_dict)
			else:
				CLIENT_MODEL_TEMPERATURE.append(data_temperature_dict)

	print("round label", ROUND_LABELS)
	print("temperature", CLIENT_MODEL_TEMPERATURE)
	print("cpu freq", CLIENT_MODEL_CPU_FREQ)

	return render_template('DevicePage.htm', TrainCurrentInfo=train_current_status, RoundLabel=ROUND_LABELS, TrainModelHeapSize=CLIENT_MODEL_HEAPSIZE, TrainModelTemperature=CLIENT_MODEL_TEMPERATURE, TrainModelCPUFrequency=CLIENT_MODEL_CPU_FREQ, WifiInfo={Nssid:Aip})

@app.route('/networkinfo', methods=['POST', 'GET'])
def networkinfo_page():
	# wifi
	Nssid, Aip = wifi_info()
	
	return render_template('NetworkPage.htm', WifiInfo={Nssid:Aip})

@app.route('/upload_h5_weight', methods=['POST'])
def upload_h5_weight_file():
	uploaded_h5_file = request.files['h5_file']
	uploaded_weight_file = request.files['weight_file']
	if uploaded_h5_file.filename != '' and uploaded_weight_file.filename != '':
		uploaded_h5_file.save('./saved_model/' + uploaded_h5_file.filename)
		uploaded_weight_file.save('./saved_model/' + uploaed_weight_file.filename)

	return redirect(url_for('modelinfo_page'))

@app.route('/modelinfo', methods=['POST', 'GET'])
def modelinfo_page():
	# wifi
	Nssid, Aip = wifi_info()

	return render_template('ModelPage.htm', WifiInfo={Nssid:Aip})

@app.route('/delete')
def client_delete():
	client_name = request.args.get('cn', 'user')
	print("request delete", client_name)
	cur = con.cursor()
	cur.execute('''DELETE FROM ClientInfo WHERE username=?''', (client_name,))
	cur.execute('''DELETE FROM DeviceInfo WHERE username=?''', (client_name,))
	cur.execute('''DELETE FROM TrainInfo WHERE username=?''', (client_name,))
	cur.execute('''DELETE FROM ClassInfo WHERE uesr_name=?''', (client_name,))
	con.commit()
	#return redirect(url_for('main_page'))

	return None

@app.route('/client_wake_up')
def wake_up_client():
	global participate_username; global wake_up_client_count
	global CHECK_CLIENT_TRAINING

	rec_client_name = request.args.get('client_name', "user")
	batteryPct = request.args.get('battery_pct', "0.0")
	isCharging = request.args.get('is_charging', "False")
	wifi_conn = request.args.get('wifi_conn', "False")
	classsize_datasize = request.args.get('classsize_datasize', 'size')
	class_size = int(classsize_datasize.split(',')[0])
	data_size = classsize_datasize.split(',')[1]
	wake_up_client_count += 1

	# insert database
	cur = con.cursor()
	cur.execute('''INSERT INTO ClientInfo VALUES (?,?,?,?)''', (rec_client_name, "N/A", 0, 0,))
	cur.execute('''INSERT INTO DeviceInfo VALUES (?,?,?,?)''', (rec_client_name, batteryPct, isCharging,wifi_conn,))
	cur.execute('''INSERT INTO ClassInfo VALUES (?,?,?)''', (rec_client_name, class_size, data_size,))
	con.commit()

	print(CHECK_CLIENT_TRAINING)
	if not CHECK_CLIENT_TRAINING:
		readyClientSids.append(rec_client_name)

		print(readyClientSids)
		print('init model')
		time.sleep(8)

		wrap_json_info = json.dumps({
		    'class_size': 5,
		    'class_list': ['book', 'laptop', 'phone', 'wash', 'water']
		})

		return wrap_json_info
	else:
		print("participate_username-", rec_client_name)
		participate_username.append(rec_client_name)
		print("check participate user -", participate_username)

		wrap_json_info = json.dumps({
		    'model_version': MODEL_VERSION,
		    'current_round': ALL_CURRENT_ROUND,
		    'state': "CTW" # current running training process wait please...
		})
		return wrap_json_info

#@app.route('/class_management')
#def management_class():
    

@app.route('/client_ready')
def ready_client():
	global readyClientSids; global participate_username; training_username
	global ALL_CURRENT_ROUND; global NUM_CLIENTS_CONTACTED_PER_ROUND
	global CHECK_CLIENT_TRAINING; global CHECK_TRAIN_TIMER

	ready_client = request.args.get('client_name', "user")
	train_size = request.args.get('train_size', '0')
	class_size = request.args.get('class_size', '0')
	#model_ver = request.args.get('model_version', "0")
	curr_round = request.args.get('current_round', "0")
	android_name = request.args.get('android_name', "0")
	print("client ready : " + ready_client + " train size : " + train_size + " model version : " + str(MODEL_VERSION) + " class size : " + class_size + " android name : " + android_name)

	if ready_client in participate_username and ready_client not in readyClientSids:
		participate_username.remove(ready_client)
		readyClientSids.append(ready_client)

	# update database
	cur = con.cursor()
	cur.execute('''UPDATE ClientInfo 
					SET androidname=?,
						trainsize=?,
						classsize=?
					WHERE username=?''', (android_name, train_size, class_size, ready_client,))
	con.commit()

	#MODEL_VERSION = model_ver
	aggregation_username.append(ready_client)
	NUM_CLIENTS_CONTACTED_PER_ROUND += 1

	#if ready_client in readyClientSids and len(readyClientSids) >= MIN_NUM_WORKERS and currentRound == 0:
	if ready_client in readyClientSids and int(curr_round) == 0:
		# participate client training process -> CHECK_CLIENT_TRAINING False >> True
		if not CHECK_TRAIN_TIMER:
			CHECK_TRAIN_TIMER = True
			t = threading.Thread(target=client_training_check_timer)
			t.start()

		print("### Check Train Round ###")
		model_loss, model_acc = trainNextRound(curr_round)

		ALL_CURRENT_ROUND = int(curr_round) + 1
		wrap_json_info = json.dumps({
		    'model_version': MODEL_VERSION,
		    'current_round': ALL_CURRENT_ROUND,
		    'max_train_round': MAX_NUM_ROUNDS,
		    'model_loss': model_loss,
		    'model_acc': model_acc,
		    'upload_url': "http://" + ip_address + ":8891/upload",
		    'model_url': "http://" + ip_address + ":8891/download"})

		print(wrap_json_info)
		training_username.append(ready_client)

		return wrap_json_info
	elif ready_client in readyClientSids and int(curr_round) == int(ALL_CURRENT_ROUND):
		print("### Check Break In Train Client ###")
		wrap_json_info = json.dumps({
				'model_version': MODEL_VERSION,
				'current_round': ALL_CURRENT_ROUND,
				'max_train_round': MAX_NUM_ROUNDS,
				'model_loss': global_model_loss,
				'model_acc': global_model_acc,
				'upload_url': "http://" + ip_address + ":8891/upload",
				'model_url': "http://" + ip_address + ":8891/download"})

		print(wrap_json_info)
		training_username.append(ready_client)

		return wrap_json_info
	else:
		return "NOT REGISTER CLIENT"

@app.route('/client_update')
def update_client():
	global clientUpdateAmount; global currentRoundClientUPdates
	global ALL_CURRENT_ROUND; global MODEL_VERSION
	global global_model_loss; global global_model_acc
	global CHECK_MODEL_UPDATE; global CHECK_CLIENT_TRAINING

	update_client = request.args.get('client_name', "user")
	current_round = request.args.get('current_round', "0")
	train_time = request.args.get('train_time', "0.0")
	heap_size = request.args.get('heapsize', "0.0")
	temperature = request.args.get('temperature', "0.0")
	cpu_freq = request.args.get('cpu_freq', "0.0")
	
	acc_f1 = request.args.get('acc_f1', "0.0")
	accuracy = acc_f1.split(',')[0]
	f1_score = acc_f1.split(',')[1]
    
	#if current_round >= 2:
        
	print(ALL_CURRENT_ROUND, current_round, NUM_CLIENTS_CONTACTED_PER_ROUND)
	if int(ALL_CURRENT_ROUND) == int(current_round):
		clientUpdateAmount += 1
		currentRoundClientUpdates.append(update_client)
		print("### client update weights ###")

		if clientUpdateAmount >= NUM_CLIENTS_CONTACTED_PER_ROUND and len(currentRoundClientUpdates) > 0:
			CHECK_MODEL_UPDATE = True
			print("### Received All Client ###")
			print("### Current client state - clientUpdateAmount: " + str(clientUpdateAmount) + ", NUM_CLIENTS_CONTACTED_PER_ROUND: " + str(NUM_CLIENTS_CONTACTED_PER_ROUND) + " ###")
			updateWeight()
			print("### update global model success ###")
			if int(current_round) >= int(MAX_NUM_ROUNDS):
				print("### finish all training!!! ###")
				eval_res = stopAndEval()

				ALL_CURRENT_ROUND = int(current_round) + 1
				MODEL_VERSION += 1
				print("change model version: " + str(MODEL_VERSION-1) + " -> " + str(MODEL_VERSION))
				wrap_json_info_eval = json.dumps({
				    'model_version': MODEL_VERSION,
				    'current_round': ALL_CURRENT_ROUND,
				    'model_loss': eval_res['loss'],
				    'model_acc': eval_res['accuracy'],
				    'state': 'FIN'})
				CHECK_MODEL_UPDATE = False

				return wrap_json_info_eval
			else:
				clientUpdateAmount -= 1

				model_loss, model_acc = trainNextRound(current_round)
				global_model_loss = model_loss
				global_model_acc = model_acc

				ALL_CURRENT_ROUND = int(current_round) + 1
				MODEL_VERSION += 1
				print("change model version: " + str(MODEL_VERSION-1) + " -> " + str(MODEL_VERSION))
				wrap_json_info = json.dumps({
				    'model_version': MODEL_VERSION,
				    'current_round': ALL_CURRENT_ROUND,
				    'model_loss': model_loss,
				    'model_acc': model_acc,
				    'upload_url': "http://" + ip_address + ":8891/upload",
				    'model_url': "http://" + ip_address + ":8891/download",
				    "state" : "RESP_ARY"})
				CHECK_MODEL_UPDATE = False
				
				# if client in traininfo database -> + acc, + loss
				print("Check client model update -> Insert to TrainInfo database")
				insert_traininfo_db(update_client, accuracy, f1_score, train_time, heap_size, cpu_freq, temperature)

				if len(participate_username) > 0 and CHECK_CLIENT_TRAINING:
					waiting_username.append(update_client)
					training_username.remove(update_client)
					time.sleep(120)
					#t1 = threading.Thread(target=next_round_train_timer)
					#t1.start(); t1.join()
					CHECK_CLIENT_TRAINING = False
				elif CHECK_CLIENT_TRAINING:
					waiting_username.append(update_client)
					training_username.remove(update_client)
					CHECK_CLIENT_TRAINING = False

				training_username.append(update_client)
				try:
					waiting_username.remove(update_client)
				except:
					print("not in waiting_username // check please")
				return wrap_json_info # "No train from all rounds yet."
        
		print("Check client model update -> Insert to TrainInfo database but now not all train")
		# accuracy, loss(x) -> f1 score (o)
		insert_traininfo_db(update_client, accuracy, f1_score, train_time, heap_size, cpu_freq, temperature)
        	
		waiting_username.append(update_client)
		training_username.remove(update_client)
		wrap_json_info = json.dumps({
		    'model_version': MODEL_VERSION,
		    'current_round': ALL_CURRENT_ROUND,
		    'state': "RESP_ACY"})

		return wrap_json_info # "No response from all clients yet."

	wrap_json_info = json.dumps({
	'model_version': MODEL_VERSION,
	'state': "NEC"})

	return wrap_json_info # "Not equal currentRound"

def insert_traininfo_db(uc, ma, ml, tt, hs, cf, te):
	cur = con.cursor()
	cur.execute('''SELECT EXISTS (SELECT * FROM TrainInfo WHERE username=?)''', (uc,))
	if cur.fetchone()[0] == 1: # exist
		cur.execute('''SELECT * FROM TrainInfo WHERE username=?''', (uc,))
		search_client_info = list(cur.fetchone())
		search_client_info[1] += "," + str(ma)
		search_client_info[2] += "," + str(ml)
		search_client_info[5] += "," + tt
		search_client_info[6] += "," + hs
		search_client_info[7] += "," + cf
		search_client_info[8] += "," + te
		cur.execute('''UPDATE TrainInfo
				SET model_acc=?,
					model_loss=?,
					current_round=?,
					model_version=?,
					training_time=?,
					heap_size=?,
					cpu_freq=?,
					temperature=?
				WHERE username=?''', (search_client_info[1], search_client_info[2], ALL_CURRENT_ROUND, MODEL_VERSION, search_client_info[5], search_client_info[6], search_client_info[7], search_client_info[8], uc,))
	else:
		print(uc + " insert to database first")
		cur.execute('''INSERT INTO TrainInfo VALUES (?,?,?,?,?,?,?,?,?)''', (uc, str(ma), str(ml), ALL_CURRENT_ROUND, MODEL_VERSION, tt, hs, cf, te,))
	con.commit()

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

	if CHECK_MODEL_UPDATE or int(model_ver) == int(MODEL_VERSION):
		return json.dumps({'state': "WAIT"})
	elif CHECK_CLIENT_TRAINING and int(model_ver) != int(MODEL_VERSION):
		

		return json.dumps({
			'class_size': 5,
			'class_list': ['book', 'laptop', 'phone', 'wash', 'water'],
			'current_round': ALL_CURRENT_ROUND,
			'model_version': MODEL_VERSION,
			'state': "NOT WAIT"})
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

		waiting_username.remove(version_client)
		if version_client not in training_username:
			training_username.append(version_client)

		return wrap_json_info

def stopAndEval():
	print("### Finish All Round ###")

	# eval
	model_evalResult = eval_model.train_model_tosave()
	print("Evaluate Loss : " + str(model_evalResult['loss']) + " Evaluate Accuracy : " + str(model_evalResult['accuracy']))

	# 이후에 각 클라이언트로 결과를 전송하기 위해서는 socket?구현이 필요한가?
	return model_evalResult

def updateWeight():
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
		layers.Conv2D(20, (5, 5), strides=(1, 1), activation='relu'),
		layers.MaxPooling2D((2, 2)),
		layers.Conv2D(50, (5, 5), strides=(1, 1), activation='relu'),
		layers.MaxPooling2D((2, 2)),
		layers.Conv2D(30, (5, 5), strides=(1, 1), activation='relu'),
		layers.Conv2D(20, (5, 5), strides=(1, 1), activation='relu'),
			layers.Dropout(0.5),
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
