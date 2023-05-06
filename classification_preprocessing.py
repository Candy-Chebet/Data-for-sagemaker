# Import necessary libraries and packages
import pandas as pd
import argparse
import os

from sklearn.model_selection import train_test_split


# Set up the argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--target', type=str, required=True, help='Specify the target variable to predict')

# Define the input and output directories
input_dir = '/opt/ml/processing/input'
train_dir = '/opt/ml/processing/train'
val_dir = '/opt/ml/processing/validation'

# Parse the input arguments
args = parser.parse_args()

# Load the data
data = pd.read_csv(os.path.join(input_dir, 'data.csv'))

# Choose the target variable to predict
target = args.target

# Split the data into training and validation sets
train, val = train_test_split(data, test_size=0.2, random_state=42)

# Create the output directories if they don't already exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Save the training and validation data to the output directories
train.to_csv(os.path.join(train_dir, 'train.csv'), index=False)
val.to_csv(os.path.join(val_dir, 'validation.csv'), index=False)
