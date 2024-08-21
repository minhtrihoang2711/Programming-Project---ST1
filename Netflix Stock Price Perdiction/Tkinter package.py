# This Function can be called from any from any front end tool/website(e.g Xboot)
import tkinter as tk
from tkinter import messagebox
from tkinter import ttk
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


#Date : 21/8/2024
#Tkinter package
#Name: Anthony
#This function is to predict more value which are all selectable And Perdict the future close price.

# Add Title and bring in the data
class StockPricePredictionApp:

    def __init__(self, master):
        self.master = master
        self.master.title('NFLX Price Prediction')
        self.data = pd.read_csv('NFLX.csv')
        self.sliders = []

        # Preprocess the data of read data in columns
        self.data = self.preprocess_data(self.data)
        self.features = ['Open','High','Low','Close']

        self.X = self.data.drop('Close', axis=1).values
        self.y = self.data['Close'].values

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        self.model = XGBRegressor()
        self.model.fit(self.X_train, self.y_train)

        self.create_widgets()

    #Drop what i don't want.
    def preprocess_data(self, data):
        # Delet 'Date', 'Adj Close', and 'Volume'
        data = data.drop(['Date', 'Adj Close', 'Volume'], axis=1, errors='ignore')

        for column in data.columns:
            if data[column].dtype == 'object':
                data = data.drop(column, axis=1)

        return data

    def create_widgets(self):
        for i, column in enumerate(self.data.columns[:-1]):
            label = tk.Label(self.master, text=column + ': ')
            label.grid(row=i, column=0)
            current_val_label = tk.Label(self.master, text='0.0')
            current_val_label.grid(row=i, column=2)
            slider = ttk.Scale(self.master, from_=self.data[column].min(), to=self.data[column].max(), orient="horizontal",
                               command=lambda val, label=current_val_label: label.config(text=f'{float(val):.2f}'))
            slider.grid(row=i, column=1)
            self.sliders.append(slider)

        predict_button = tk.Button(self.master, text='Predict Price', command=self.predict_price)
        predict_button.grid(row=len(self.data.columns[:-1]), columnspan=3)

    def predict_price(self):
        inputs = [float(slider.get()) for slider in self.sliders]
        price = self.model.predict([inputs])
        messagebox.showinfo('Predicted Price', f'The predicted NFLX Close stock price is ${price[0]:.2f}')

if __name__ == '__main__':
    root = tk.Tk()
    app = StockPricePredictionApp(root)
    root.mainloop()
