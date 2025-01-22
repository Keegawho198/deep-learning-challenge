# deep-learning-challenge

Run code using colob in google drive

## Overview
This project aims to develop a machine learning model to predict the success of funding applicants for the nonprofit organization Alphabet Soup. Using a dataset of past applicants, we built and trained a deep learning model to classify successful applications.

## Technologies Used
- Python
- Pandas
- TensorFlow
- Scikit-learn

## Files Included
- `charity_data.csv`: The dataset used for the analysis.
- `AlphabetSoupCharity.h5`: The trained neural network model.
- `AlphabetSoupCharity.ipynb`: The Jupyter Notebook containing the analysis code.
- `README.md`: Project overview and instructions.

## Methodology
1. **Data Preprocessing:**
   - Removed unnecessary columns (`EIN`, `NAME`).
   - Encoded categorical variables using `pd.get_dummies`.
   - Scaled numerical features using `StandardScaler`.
   - Split the data into training and testing sets.

2. **Model Building:**
   - Designed a deep neural network with two hidden layers and ReLU activation.
   - Output layer used sigmoid activation for binary classification.

3. **Model Training and Evaluation:**
   - Trained the model for 100 epochs.
   - Evaluated performance using accuracy on the test dataset.

## Results
- The model achieved a test accuracy of approximately 73%.

## Recommendations
- Consider using alternative machine learning models like Random Forest or Gradient Boosting.
- Apply feature selection techniques to refine input features.
- Optimize hyperparameters using cross-validation.

## Usage
To use the model:
1. Clone this repository.
2. Run the Jupyter Notebook `AlphabetSoupCharity.ipynb`.
3. Load the model using the provided HDF5 file for further predictions.