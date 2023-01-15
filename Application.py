# load all the required libraries
from flask import Flask,request
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
# create object
app = Flask(__name__)


# create end point to  train your model and save training data in pickle file
@app.route('/train_model')
def train():
    # load data
    data = pd.read_excel('Historical Alarm Cases.xlsx')
    # split columns
    x = data.iloc[:, 1:7]
    y = data['Spuriosity Index(0/1)']
    # create object for Algo class
    logm = LogisticRegression()
    # train the model
    logm.fit(x, y)
    # Save trainig results in pickle file
    joblib.dump(logm, 'train.pkl')
    return "Model trained successfully"


#  load pickle file and test your model, pass test data via POSt method
#  First we need to load pickle file for it to get training data ref
@app.route('/test_model', methods=['POST'])
def test():
    # load pickle file
    pkl_file = joblib.load('train.pkl')
    test_data = request.get_json()
    f1 = test_data['Ambient Temperature( deg C)']
    f2 = test_data['Calibration(days)']
    f3 = test_data['Unwanted substance deposition(0/1)']
    f4 = test_data['Humidity(%)']
    f5 = test_data['H2S Content(ppm)']
    f6 = test_data['detected by(% of sensors)']
    my_test_data = [f1, f2, f3, f4, f5, f6]
    my_data_array = np.array(my_test_data)
    test_array = my_data_array.reshape(1, 6)
    df_test = pd.DataFrame(test_array,
                           columns=['Ambient Temperature( deg C)', 'Calibration(days)', 'Unwanted substance '
                                                                                        'deposition(0/1)',
                                    'Humidity(%)',
                                    'H2S Content(ppm)', 'detected by(% of sensors)'])
    y_pred = pkl_file.predict(df_test)

    if y_pred == 1:
        return "False Alarm, No Danger"
    else:
        return "True Alarm, Danger "


#  run the application on port
app.run(port=6000)
