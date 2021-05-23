
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler

st.title('📈 STOCK MARKET ANALYSIS AND PREDICTION USING NEURAL NETWORKS 📉')

def format_func(option):
    return CHOICES[option]
CHOICES = {1: "ADBL", 2: "EBL", 3: "HBL", 4: "KBL"}
option = st.selectbox("Select option", options=list(CHOICES.keys()), format_func=format_func)
print(option)
st.write(f"You selected {format_func(option)}")
bank = format_func(option)
if st.button("ANALYZE"):
        if bank == "ADBL":
            regressor = load_model("models\ADBL.h5")
            dataset_test = pd.read_csv('traning\ADBL_test_data.csv')
            dataset_train = pd.read_csv('traning\ADBL_train_data.csv')
            training_set=dataset_train.iloc[:,5:6].values
            sc = MinMaxScaler(feature_range=(0,1))
            training_set_scaled=sc.fit_transform(training_set)
            real_stock_price = dataset_test.iloc[:,5:6].values
            dataset_total = pd.concat((dataset_train['Close'],dataset_test['Close']),axis = 0)
            inputs = dataset_total[len(dataset_total)-len(dataset_test)-60:].values
            inputs = inputs.reshape(-1,1)
            inputs = sc.transform(inputs)
            X_test = []
            for i in range(60,80):
                X_test.append(inputs[i-60:i,0])
            X_test=np.array(X_test)
            X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
            predicted_stock_price = regressor.predict(X_test)
            predicted_stock_price = sc.inverse_transform(predicted_stock_price)
            plt.plot(real_stock_price, color = 'red', label = 'Real ADBL Stock Price')
            plt.plot(predicted_stock_price, color = 'blue', label = 'Predicted ADBL Stock Price')
            plt.title('ADBL Stock Price Prediction')
            plt.xlabel('Time')
            plt.ylabel('ADBL stock Price')
            plt.legend()
            # plt.show()
            plt.savefig('static\\temp.png')
            image = Image.open("static\\temp.png")
            st.image(image, use_column_width=True)
        elif bank == "EBL":
            mu, sigma = 100, 15
            x = mu + sigma * np.random.randn(10000)
            # the histogram of the data
            n, bins, patches = plt.hist(x, 50, density=1, facecolor='g', alpha=0.75)
            plt.xlabel('Smarts')
            plt.ylabel('Probability')
            plt.title('Histogram of IQ')
            plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
            plt.axis([40, 160, 0, 0.03])
            plt.grid(True)
            #plt.show()
            plt.savefig('static\\temp.png')
            image = Image.open("static\\temp.png")
            st.image(image, use_column_width=True)



   


