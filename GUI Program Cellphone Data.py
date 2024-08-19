from re import IGNORECASE
import pickle
import pandas as pd

def FunctionPredictResult(InputData):
    Num_Inputs=InputData.shape[0]

    # Making sure the input data has same columns as it was used for training the model
    # Also, if standardization/normalization was done, then same must be done for new input

    # Appending the new data with the Training data
    DataForML=pd.read_pickle('/Users/minhtrihoang/Developer/Software Technology/Group Assignment/DataForML.pkl')
    #InputData=InputData.append(DataForML, ignore_index=True)
    InputData = pd.concat([InputData, DataForML], ignore_index=True)

    # Generating dummy variables for rest of the nominal variables
    InputData=pd.get_dummies(InputData)

    # Maintaining the same order of columns as it was during the model training
    Predictors=['cpu core', 'cpu freq', 'internal mem','ram']

    # Generating the input values to the model
    X=InputData[Predictors].values[0:Num_Inputs]

    with open('/Users/minhtrihoang/Developer/Software Technology/Group Assignment/Final_Linear_Model.pkl', 'rb') as fileReadStream:
        PredictionModel=pickle.load(fileReadStream)
        # Close the filestream!
        fileReadStream.close()

    # Genrating Predictions
    Prediction=PredictionModel.predict(X)
    PredictionResult=pd.DataFrame(Prediction, columns=['Prediction'])
    return(PredictionResult)

# Calling the function for some new data
NewSampleData=pd.DataFrame(data=[[8,2.3,16,3],[2,1.3,4,0.512]],columns=['cpu core', 'cpu freq', 'internal mem','ram'])

print(NewSampleData)

# Calling the Function for prediction
FunctionPredictResult(InputData=NewSampleData)
print(FunctionPredictResult(InputData=NewSampleData))

