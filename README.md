Customer churn dataset and prediction of the churn using ANN


Overview:
This project involves building an Artificial Neural Network (ANN) for predicting customer churn. The dataset used contains various customer attributes, and the ANN is trained to predict whether a customer is likely to leave the bank.

Files:
artificial_neural_network(Customer Churn Prediction).ipynb: Jupyter Notebook containing the code for data preprocessing, model building, training, and evaluation.
Churn_Modelling.csv: Dataset used for training and testing the ANN.

Workflow:
Importing Libraries: Necessary libraries such as NumPy, Pandas, TensorFlow, and Keras are imported.
Data Preprocessing: The dataset is loaded, and data preprocessing steps include handling categorical data, label encoding, one-hot encoding, splitting the dataset, and feature scaling.
Building the ANN: A Sequential model is created using TensorFlow and Keras. The model architecture consists of an input layer, two hidden layers with ReLU activation, and an output layer with sigmoid activation.
Training the ANN: The model is compiled using the Adam optimizer and binary crossentropy loss. It is then trained on the training set for 100 epochs.
Making Predictions and Evaluating the Model: Predictions are made on the test set, and the model's performance is evaluated using a confusion matrix and accuracy score.

Results
Accuracy: The trained model achieves an accuracy of approximately 86.3% on the test set.

Prediction Example:
An example is provided where the model predicts whether a customer with specific attributes will leave the bank. The model predicts that the customer stays.

Google Colab:
Open the Jupyter Notebook in Google Colab by clicking on artificial_neural_network(Customer_Churn_Prediction).ipynb.

Execute each cell in order.

Local Environment:
Download the Jupyter Notebook artificial_neural_network(Customer_Churn_Prediction).ipynb and open it in a Jupyter Notebook environment with the required dependencies.


