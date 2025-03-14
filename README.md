EMNIST Letter Classification using CNN and ViT

Overview

This project implements and compares Convolutional Neural Networks (CNN) and Vision Transformer (ViT) models for detecting and classifying handwritten letters in the EMNIST dataset. The models used include:

CNN: A deep learning model that extracts spatial features through convolutional layers.

ViTForImageClassification: A transformer-based vision model designed for image classification.

ViTFeatureExtractor: Used for preprocessing images before feeding them into the ViT model.

Dataset

The EMNIST (Extended MNIST) dataset is a set of handwritten character datasets that serve as an extension of the original MNIST dataset. This project specifically utilizes the EMNIST Letters split, which consists of 26 classes (A-Z).

Dataset source: EMNIST Dataset

Project Structure

├── dataset/                     # Contains EMNIST data (if downloaded manually)
├── models/                      # Saved trained models
├── notebooks/                   # Jupyter notebooks for model training and evaluation
├── src/                         # Source code
│   ├── cnn_model.py             # CNN model implementation
│   ├── vit_model.py             # ViT model implementation
│   ├── train.py                 # Training script for both models
│   ├── evaluate.py              # Evaluation script
│   ├── preprocess.py            # Preprocessing functions for the dataset
├── requirements.txt             # Required dependencies
├── README.md                    # Project documentation

Installation

Clone the repository:

git clone https://github.com/your-username/emnist-letter-classification.git
cd emnist-letter-classification

Create a virtual environment (optional but recommended):

python -m venv env
source env/bin/activate  # On Windows use: env\Scripts\activate

Install dependencies:

pip install -r requirements.txt

Model Training

Run the training script to train both models:

python src/train.py --model cnn   # Train CNN model
python src/train.py --model vit   # Train ViT model

Evaluation

Evaluate the trained models on the test set:

python src/evaluate.py --model cnn   # Evaluate CNN model
python src/evaluate.py --model vit   # Evaluate ViT model

Results

The model performances are compared based on accuracy, loss, and computational efficiency. Results and visualizations are available in the Jupyter notebooks under notebooks/.

Future Improvements

Experimenting with different ViT architectures.

Hyperparameter tuning for better accuracy.

Data augmentation techniques to improve generalization.

Using additional transformer-based models for comparison.

Contributing

Feel free to contribute by opening issues or submitting pull requests!

License

This project is licensed under the MIT License.
