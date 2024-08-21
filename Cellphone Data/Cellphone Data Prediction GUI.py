import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

# Class definition for the Cellphone Price Prediction Application
class HousePricePredictionApp:
    def __init__(self, master):
        self.master = master
        self.master.title('Cellphone Price Prediction')  # Name the window title
        # Load the dataset
        self.data = pd.read_csv('/Users/minhtrihoang/Developer/Software Technology/Group Assignment/Cellphone.csv')
        self.sliders = []  # List to hold slider widgets

        # Prepare the data for training
        self.X = self.data.drop('Price', axis=1).values  # Features
        self.y = self.data['Price'].values  # Target variable

        # Split the data into training and testing sets
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Initialize and train the linear regression model
        self.model = LinearRegression()
        self.model.fit(self.X_train, self.y_train)

        # Create the GUI widgets
        self.create_widgets()

    def create_widgets(self):
        # Create sliders for each feature in the dataset
        for i, column in enumerate(self.data.columns[:-1]):
            label = tk.Label(self.master, text=column + ': ')  # Label for the feature
            label.grid(row=i, column=0)
            current_val_label = tk.Label(self.master, text='0.0')  # Label to display current slider value
            current_val_label.grid(row=i, column=2)
            # Slider for the feature
            slider = ttk.Scale(self.master, from_=self.data[column].min(), to=self.data[column].max(), orient="horizontal",
                               command=lambda val, label=current_val_label: label.config(text=f'{float(val):.2f}'))
            slider.grid(row=i, column=1)
            self.sliders.append((slider, current_val_label))  # Add slider and label to the list

        # Button to trigger price prediction
        predict_button = tk.Button(self.master, text='Predict Price', command=self.predict_price)
        predict_button.grid(row=len(self.data.columns[:-1]), columnspan=3)

    def predict_price(self):
        # Get the values from the sliders
        inputs = [float(slider.get()) for slider, _ in self.sliders]
        # Predict the price using the trained model
        price = self.model.predict([inputs])
        # Show the predicted price in a message box
        messagebox.showinfo('Predicted Price', f'The predicted phone price is ${price[0]:.2f}')

if __name__ == '__main__':
    root = tk.Tk()  # Create the main window
    app = HousePricePredictionApp(root)  # Create an instance of the application
    root.mainloop()  # Start the Tkinter event loop