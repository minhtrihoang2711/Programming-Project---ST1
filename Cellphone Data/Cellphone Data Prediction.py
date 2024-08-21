from re import IGNORECASE
import pickle
import pandas as pd

# Function to predict results based on input data
def FunctionPredictResult(InputData):
    # Get the number of input samples
    Num_Inputs = InputData.shape[0]

    # Load the preprocessed data for machine learning
    DataForML = pd.read_pickle('/Users/minhtrihoang/Developer/Software Technology/Group Assignment/DataForML.pkl')
    
    # Concatenate the input data with the preprocessed data
    InputData = pd.concat([InputData, DataForML], ignore_index=True)

    # Convert categorical variables into dummy/indicator variables
    InputData = pd.get_dummies(InputData)

    # Define the predictors (features) to be used for prediction
    Predictors = ['cpu core', 'cpu freq', 'internal mem', 'ram']

    # Extract the feature values for the input samples
    X = InputData[Predictors].values[0:Num_Inputs]

    # Load the pre-trained prediction model from a pickle file
    with open('/Users/minhtrihoang/Developer/Software Technology/Group Assignment/Final_Linear_Model.pkl', 'rb') as fileReadStream:
        PredictionModel = pickle.load(fileReadStream)
        fileReadStream.close()

    # Use the model to predict the output based on the input features
    Prediction = PredictionModel.predict(X)
    
    # Create a DataFrame to store the prediction results
    PredictionResult = pd.DataFrame(Prediction, columns=['Prediction'])
    
    # Return the prediction results
    return PredictionResult

# Create a new sample data for prediction
NewSampleData = pd.DataFrame(data=[[8, 2.3, 16, 3], [2, 1.3, 4, 0.512]], columns=['cpu core', 'cpu freq', 'internal mem', 'ram'])

# Print the new sample data
print(NewSampleData)

# Call the prediction function and print the results
print(FunctionPredictResult(InputData=NewSampleData))

