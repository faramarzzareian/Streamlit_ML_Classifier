# Streamlit_ML_Classifier

This is a simple machine learning GUI built using the Streamlit framework and the popular machine learning libraries like NumPy, Pandas and Scikit-learn.

The GUI enables the user to select a dataset either from "Wine dataset" or "Breast Cancer" and a classifier either "KNN" or "SVM". The parameters for each classifier can be set from the sidebar. The classifier is then trained on the selected dataset and the accuracy score is displayed.
Installation

## To install the required libraries, run the following command in your terminal:

pip install streamlit numpy pandas scikit-learn

## Usage

To run the GUI, execute the following command in your terminal:

streamlit run file_name.py

## Customization

This code can be easily extended to include more datasets and classifiers. To add a new dataset, modify the get_data function and add the necessary code to load the data. To add a new classifier, modify the get_classifier function and add the necessary code to instantiate the classifier. The parameters for the new classifier can be set in the add_p function.
