#Date : 21/8/2024
#Create a python function
#Name: Anthony
#This function is to predict the what will the future price be.
# This Function can be called from any from any front end tool/website(e.g Xboot)
import pandas as pd
from re import IGNORECASE
from xgboost import XGBRegressor


def FunctionPredictResult(InputData):
    import pandas as pd
    Num_Inputs=InputData.shape[0]

    # Appending the new data with the Training data
    DataForML=pd.read_pickle('DataForMLNetflixData.pkl')
    #InputData=InputData.append(DataForML, ignore_index=True)
    InputData = pd.concat([InputData, DataForML], ignore_index=True)

    # Generating dummy variables for rest of the nominal variables
    InputData=pd.get_dummies(InputData)

    # Maintaining the same order of columns as it was during the model training
    Predictors=['Open','High','Low']

    # Generating the input values to the model
    X=InputData[Predictors].values[0:Num_Inputs]



    # Loading the Function from pickle file
    import pickle

    with open('Final_XGB_ModelNetflixData.pkl', 'rb') as fileReadStream:
        PredictionModel=pickle.load(fileReadStream)
        # Close the filestream!
        fileReadStream.close()

    # Genrating Predictions
    Prediction=PredictionModel.predict(X)
    PredictionResult=pd.DataFrame(Prediction, columns=['Prediction'])
    return(PredictionResult)


# Calling the function for some new data
NewNetflixData=pd.DataFrame(data=[[262,267.899994,250.029999],[247.699997,266.700012,245]],columns=('Open','High','Low'))

print(NewNetflixData)

# Calling the Function for prediction and print.
FunctionPredictResult(InputData=NewNetflixData)
print(FunctionPredictResult(InputData=NewNetflixData))