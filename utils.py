import numpy as np
import pandas as pan
import itertools
import matplotlib.pyplot as plt
import os
import time
from sklearn.utils import shuffle

#Method for generating a confidence matrix plot at the end
def make_confMat_plot(cm,title="Confusion Matrix", cmap=plt.cm.Blues):
	classes = [1,2,3,4,5,6,7,8,9,10]
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	cm = np.round(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis],decimals = 2)
	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")
	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

#Method for loading training data
def load_my_train_data(data_type='struct_test', data_set = '1'):
	if data_type == 'struct_test':
		x_test_train_array = "../Raw_Data/small_lists/test_training_xs.csv"
		y_test_train_array = "../Raw_Data/small_lists/test_training_ys.csv"
		X = np.loadtxt(open(x_test_train_array, "rb"), delimiter = ",", skiprows=0)
		Y = np.loadtxt(open(y_test_train_array, "rb"), delimiter = ",", skiprows=0)
	elif data_type == 'final':
		final_train_x = "../Raw_Data/flipped_data/flipped_train.csv"
		final_train_ans = "../Raw_Data/flipped_data/train_answers.csv"
		X = np.loadtxt(open(final_train_x, "rb"), delimiter = ",", skiprows=0)
		Y = np.loadtxt(open(final_train_ans, "rb"), delimiter = ",", skiprows=0)
	elif data_type == 'test_short':
		data_loc1 = '../Raw_Data/cross_val_sets/cross'
		data_loc2 = '_sample.csv'
		X = np.loadtxt(open(data_loc1+data_set+data_loc2, "rb"), delimiter = ",", skiprows=0)
		Y = np.loadtxt(open(data_loc1+data_set+'_sample_ans.csv', "rb"), delimiter = ",", skiprows=0)
		X, Y = shuffle(X, Y)
	elif data_type =='test_long':
		final_train_x = "dataset/flipped_train.csv"
		final_train_ans = "dataset/train_answers.csv"
		X = np.loadtxt(open(final_train_x, "rb"), delimiter = ",", skiprows=0)
		Y = np.loadtxt(open(final_train_ans, "rb"), delimiter = ",", skiprows=0)
	return X,Y

def load_my_test_data(data_type='struct_test', data_set='1', dat_sz = 's'):
	if data_type == 'struct_test':
		x_test_test_array = "../Raw_Data/small_lists/test_testing_xs.csv"
		y_test_test_array = "../Raw_Data/small_lists/test_testing_ys.csv"
		test_xs = np.loadtxt(open(x_test_test_array, "rb"), delimiter = ",", skiprows=0)
		test_ys = np.loadtxt(open(y_test_test_array, "rb"), delimiter = ",", skiprows=0)
	elif data_type == 'final':
		if dat_sz == 's':
			final_test_data = "../Raw_Data/flipped_data/test_sets/stest_set_"
		elif dat_sz == 'm':
			final_test_data = "../Raw_Data/flipped_data/test_sets/test_set_"
		test_xs = np.loadtxt(open(final_test_data+data_set+'.csv', "rb"), delimiter = ",", skiprows=0)
		test_ys = 0
	elif data_type == 'test_short':
		data_loc1 = '../Raw_Data/cross_val_sets/cross'
		data_loc2 = '_sample.csv'
		test_xs = np.loadtxt(open(data_loc1+data_set+data_loc2, "rb"), delimiter = ",", skiprows=0)
		test_ys = np.loadtxt(open(data_loc1+data_set+'_sample_ans.csv', "rb"), delimiter = ",", skiprows=0)

	return test_xs, test_ys

def create_output_dir(class_name):
	date = time.strftime("%m_%d")
	file_path = class_name+'_'+date
	if not os.path.exists(file_path):
		os.makedirs(file_path)
	return file_path+'/'

#def test_classifier(classifier)
