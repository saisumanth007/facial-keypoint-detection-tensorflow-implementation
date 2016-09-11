import matplotlib.pyplot as plt
from sklearn.utils import shuffle
import pandas as pd
import numpy as np
from sklearn.externals import joblib
from sklearn.learning_curve import learning_curve
import os

filedir = os.getcwd()

ftrain = filedir+'/data/training.csv'
ftest = filedir+'/data/test.csv'
flookup = filedir+'/data/IdLookupTable.csv'

BATCH_SIZE = 64
IMAGE_SIZE = 96
LABELS_SIZE = 30
VALIDATION_SIZE = 100
NUM_EPOCHS = 1000
SEED = None
EVAL_BATCH_SIZE = 64
EARLY_STOP_PATIENCE = 100
NUM_CHANNELS = 1

def load_data(test = False, cols = None):
	"""Helper function which loads the data from train file or test file based
	on the arguments.
	
	Args:
	
	test (Optional): Specifies which data to load i.e., train or test. 
	cols (Optional[list]): List of specific columns of interest in loaded dataframe.

	Returns:

	Returns loaded data after manipulation as required.
	
	"""
	file_name = ftest if test else ftrain
	df = pd.read_csv(file_name) #loading the pandas dataframe

	# Store the values of Image Column in an numpy array
	df['Image'] = df['Image'].apply(lambda i: np.fromstring(i, sep=' '))

	#If cols argument given:
	#Retrieve the respective columns and update the dataframe df
	if cols:
		df = df[list(cols)+['Image']]

	#Drop all rows with null values
	df = df.dropna()

	#Rescale the values of Image column to [0, 1]
	X = np.vstack(df['Image'].values)/255
	X = X.astype(np.float32)
	X = X.reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 1)
	
	if not test:	# Only Training Data has the target columns

		y = df[df.columns[:-1]].values / 96.0	#Rescale the values in target columns to [-1, 1]
		X, y = shuffle(X, y, random_state = 42)
		joblib.dump(cols, 'data/cols.pkl', compress=3) # Shuffle the training data
		y = y.astype(np.float32)
	else:
		y = None

	return X, y

def eval_in_batches(data, sess, eval_prediction, eval_data_node):
    """Get all predictions for a dataset by running it in small batches."""
    size = data.shape[0]
    if size < EVAL_BATCH_SIZE:
        raise ValueError("batch size for evals larger than dataset: %d" % size)
    predictions = np.ndarray(shape=(size, LABELS_SIZE), dtype=np.float32)
    for begin in xrange(0, size, EVAL_BATCH_SIZE):
        end = begin + EVAL_BATCH_SIZE
        if end <= size:
            predictions[begin:end, :] = sess.run(
                eval_prediction,
                feed_dict={eval_data_node: data[begin:end, ...]})
        else:
            batch_predictions = sess.run(
                eval_prediction,
                feed_dict={eval_data_node: data[-EVAL_BATCH_SIZE:, ...]})
            predictions[begin:, :] = batch_predictions[begin - size:, :]
	return predictions


def error_measure(predictions, labels):
	return np.sum(np.power(predictions - labels, 2)) / (2 * predictions.shape[0])

def generate_submission(test_dataset, sess, eval_prediction, eval_data_node):
    test_labels = eval_in_batches(test_dataset, sess, eval_prediction, eval_data_node)
    test_labels *= 96.0
    test_labels = test_labels.clip(0, 96)

    lookup_table = pd.read_csv(FLOOKUP)
    values = []

    cols = joblib.load('data/cols.pkl')

    for index, row in lookup_table.iterrows():
        values.append((
            row['RowId'],
            test_labels[row.ImageId - 1][np.where(cols == row.FeatureName)[0][0]],
        ))
    submission = pd.DataFrame(values, columns=('RowId', 'Location'))
    submission.to_csv('data/submission.csv', index=False)
