from data_preparation import load_and_process_data

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Dropout
from tensorflow.keras.callbacks import EarlyStopping
import numpy as np

# Load and process the data
data_path = 'E:/PHD Projects/Project 1/mitdb'
data, labels = load_and_process_data(data_path)