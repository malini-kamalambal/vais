# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import argparse
import logging
import os

import numpy as np
import pandas as pd
import pathlib
import tensorflow as tf

from datetime import datetime
from sklearn.ensemble import IsolationForest
import joblib

print(tf.__version__)

# If the input CSV file has a header row, then set CSV_COLUMNS to None.
# Otherwise, set CSV_COLUMNS to a list of target and feature names:
# i.e. CSV_COLUMNS = None
CSV_COLUMNS = [
    'dimension_1',
    'dimension_2'
]

# Target name
# i.e. TARGET_NAME = 'tip'
TARGET_NAME = None

# The features to be used for training.
# If FEATURE_NAMES is None, then all the available columns will be
# used as features, except for the target column.
# i.e. FEATURE_NAMES = ['trip_miles','trip_seconds','fare','trip_start_month','trip_start_hour','trip_start_day',]
FEATURE_NAMES = None

# If the model is serialized using joblib
# then use 'model.joblib' for the model name
MODEL_FILE_NAME = 'model.joblib'

PRED_FILE_NAME = 'prediction.results'
PRED_FILE_NAME_ERROR = 'prediction.error_stats'

# Set to True if you want to tune some hyperparameters
HYPERPARAMTER_TUNING = False

def read_df_from_gcs(file_pattern):
    """Read data from Google Cloud Storage, split into train and validation sets
    Assume that the data on GCS is in csv format without header.
    The column names will be provided through metadata
    Args:
      file_pattern: (string) pattern of the files containing training data.
      For example: [gs://bucket/folder_name/prefix]
    Returns:
      pandas.DataFrame
    """

    # Download the files to local /tmp/ folder
    df_list = []

    for filepath in tf.io.gfile.glob(file_pattern):
        with tf.io.gfile.GFile(filepath, 'r') as f:
            if CSV_COLUMNS is None:
                df_list.append(pd.read_csv(f))
            else:
                df_list.append(pd.read_csv(f, names=CSV_COLUMNS,
                                           header=None))

    data_df = pd.concat(df_list)

    return data_df

def upload_to_gcs(local_path, gcs_path):
    """Upload local file to Google Cloud Storage.
    Args:
      local_path: (string) Local file
      gcs_path: (string) Google Cloud Storage destination
    Returns:
      None
    """
    tf.io.gfile.copy(local_path, gcs_path)
    
def dump_object(object_to_dump, output_path):
    """Pickle the object and save to the output_path.
    Args:
      object_to_dump: Python object to be pickled
      output_path: (string) output path which can be Google Cloud Storage
    Returns:
      None
    """

    if not tf.io.gfile.exists(output_path):
        tf.io.gfile.makedirs(os.path.dirname(output_path))
    with tf.io.gfile.GFile(output_path, 'w') as wf:
        joblib.dump(object_to_dump, wf)

def load_object(gcs_input_path):
    """Load the object from the input_path.
    Args:
      object_to_dump: Python object to be pickled
      output_path: (string) output path which can be Google Cloud Storage
    Returns:
      None
    """

    loaded_model = joblib.load(tf.io.gfile.GFile(gcs_input_path, 'rb'))
    return loaded_model
        
def get_estimator(arguments):
    """Create an Isolation Forest classifier for anomaly detection 
    # Generate ML Pipeline which include both pre-processing and model training
    
    Args:
      arguments: (argparse.ArgumentParser), parameters passed from command-line
    Returns:
      classifier - the Isolation Forests classifier(still needs to be trained)
    """

    # max_samples and random_state_seed are expected to be passed as
    # command line argument to task.py
    
    # max_samples: “auto”, int or float, default=”auto”
    # The number of samples to draw from X to train each base estimator.
    
    # random_stateint, RandomState instance or None, default=None
    # Controls the pseudo-randomness of the selection of the feature and split values for each branching step and each tree in the forest.
    
    estimator = IsolationForest(
        max_samples=arguments.max_samples,
        random_state=arguments.random_state_seed)

    return estimator

def train_and_evaluate(estimator, dataset, output_dir):
    """Runs model training and evaluation.
    Args:
      estimator: (pipeline.Pipeline), Pipeline instance, assemble pre-processing
        steps and model training
      dataset: (pandas.DataFrame), DataFrame containing training data
      output_dir: (string), directory that the trained model will be exported
    Returns:
      None
    """
    #x_train, y_train, x_val, y_val = util.data_train_test_split(dataset)
    x_train = dataset

    estimator.fit(x_train)

    # Write model and eval metrics to `output_dir`
    model_output_path = os.path.join(output_dir,
                                     MODEL_FILE_NAME)

    dump_object(estimator, model_output_path)
    
def run_experiment(arguments):
    """Testbed for running model training and evaluation."""
    # Get data for training and evaluation

    logging.info('Arguments: %s', arguments)
    
    # Get the training data
    logging.info('Getting the training data from: ' + arguments.input)
    dataset_df = read_df_from_gcs(arguments.input)
    dataset = dataset_df.to_numpy()

    # Get estimator
    estimator = get_estimator(arguments)

    # Run training and evaluation
    logging.info('Running training, outputting model to: ' + arguments.job_dir)
    train_and_evaluate(estimator, dataset, arguments.job_dir)

def run_prediction(arguments):
    """Running model inference."""
    # Get data for training and evaluation

    logging.info('Arguments: %s', arguments)
    
    # Get the training data
    logging.info('Getting the data to predict from: ' + arguments.input)
    dataset_df = read_df_from_gcs(arguments.input)
    dataset = dataset_df.to_numpy()
    
    # Load model from `artifact_uri`
    model_path = os.path.join(arguments.artifact_uri,
                              MODEL_FILE_NAME)
    logging.info('Getting the model from: ' + model_path)
    model = load_object(model_path)

    # Generate output file path
    output_file = PRED_FILE_NAME + '.csv'
    output_path = os.path.join(arguments.gcs_destination_prefix,
                              output_file)
    
    # Run predictions
    logging.info('Running predictions, outputting scores to: ' + output_path)
    y_pred_dataset = model.predict(dataset)
    
    # Save numpy array predictions to GCS
    pd.DataFrame(y_pred_dataset).to_csv(output_path, header=None, index=None)
    
def parse_args():
    """Parses command-line arguments."""
    """Argument parser.

    Returns:
      Dictionary of arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--log-level', help='Logging level.', choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'], default='INFO')
    parser.add_argument('--input', help='CSV file to use for predictions.', required=True)
    #parser.add_argument('--job-dir', help='Output directory for exporting model and other metadata.', required=True)
    #parser.add_argument('--max-samples', type=int, default=100, help='maximum number of random samples to generate, default=100')
    #parser.add_argument('--random-state-seed', type=int, default=42, help='random seed used to initialize the pseudo-random number generator, default=42')
    #parser.add_argument('--n-estimators', default=10, type=int, help='Number of trees in the forest.')
    #parser.add_argument('--max-depth', type=int, default=3, help='The maximum depth of the tree.')
    parser.add_argument('--artifact-uri', help='Cloud Storage path to the directory that contains your model artifacts.')
    #parser.add_argument('--gcs-source', help='Google Cloud Storage URI(-s) to your instances to run batch prediction on.')
    parser.add_argument('--gcs-destination-prefix', help='The Google Cloud Storage location of the directory where the output is to be written to.')

    return parser.parse_args()

if __name__ == '__main__':
    """Entry point"""

    arguments = parse_args()
    logging.basicConfig(level=arguments.log_level)
    # Run the prediction method
    time_start = datetime.utcnow()
    run_prediction(arguments)
    time_end = datetime.utcnow()
    time_elapsed = time_end - time_start
    logging.info('Prediction elapsed time: {} seconds'.format(
        time_elapsed.total_seconds()))