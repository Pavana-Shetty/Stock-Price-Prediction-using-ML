import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np
from datetime import timedelta
import requests  # To send file via Telegram

# Telegram Bot API Configurations
TELEGRAM_BOT_TOKEN = ""  # Replace with your bot token
CHAT_ID = ""  # Replace with your chat ID or group ID

# Streamlit App Title
st.title("Stock Price Prediction App")

# Sidebar for Stock Selection
st.sidebar.header("Stock Selection")
stock_name = st.sidebar.selectbox("Select a Stock", ["INFY", "TATA", "TCS", "WIPRO", "TITAN"], index=0)

# Map stock names to file names
stock_files = {
    "INFY": "INFY.NS.csv",
    "TATA": "TATASTEEL.NS.csv",
    "TCS": "TCS.NS.csv",
    "WIPRO": "WIPRO.NS.csv",
    "TITAN": "TITAN.NS.csv",  # Replace with your actual file paths
}

# Load the selected stock data file
file_path = stock_files.get(stock_name)

try:
    data = pd.read_csv(file_path, delimiter=",")  # Adjust delimiter if necessary
except Exception as e:
    st.error(f"Error loading file for {stock_name}: {e}")
    st.stop()

# Preprocess the data
data['Date'] = pd.to_datetime(data['Date'])
data = data.sort_values('Date')
data = data.dropna()

# User Input for Prediction Horizon
st.sidebar.header("Prediction Horizon")
days_to_predict = st.sidebar.number_input("Enter number of days to predict", min_value=1, max_value=30, value=10)

# Show data summary
st.write(f"### {stock_name} Stock Data")
st.write(data.head())

# Plot historical data
st.write("### Historical Closing Prices")
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(data['Date'], data['Close'], label='Close Price')
ax.set_xlabel('Date')
ax.set_ylabel('Close Price')
ax.set_title(f'{stock_name} Close Price Over Time')
ax.legend()
st.pyplot(fig)

# Prepare features and target
X = data[['Open', 'High', 'Low', 'Volume']]
y = data['Close']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write("### Model Performance")
st.write(f"Mean Squared Error: {mse:.2f}")
st.write(f"Mean Absolute Error: {mae:.2f}")
st.write(f"R2 Score: {r2:.2f}")

# Future Predictions
if st.sidebar.button("Predict"):
    # Generate future dates
    last_date = data['Date'].iloc[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, days_to_predict + 1)]

    # Generate future feature values with noise
    last_known_features = X.iloc[-1].values
    noise_factor = 0.01
    future_features = [
        last_known_features * (1 + noise_factor * np.random.randn(4)) for _ in range(days_to_predict)
    ]
    future_features = np.array(future_features)

    # Predict future prices
    future_predictions = model.predict(future_features)

    # Store predictions in a DataFrame
    prediction_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Close Price": future_predictions
    })

    # Save predictions to a file
    prediction_file = "predicted_prices.csv"
    prediction_df.to_csv(prediction_file, index=False)

    # Display predictions
    st.write("### Predicted Prices")
    st.write(prediction_df)

    # Plot future predictions
    st.write("### Future Predictions")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(data['Date'], data['Close'], label='Historical Close Prices')
    ax.plot(future_dates, future_predictions, label='Future Predicted Prices', marker='o', color='orange')
    ax.set_xlabel('Date')
    ax.set_ylabel('Close Price')
    ax.set_title(f'{stock_name} Price Prediction for Next {days_to_predict} Days')
    ax.legend()
    st.pyplot(fig)

    # Send the file via Telegram
    st.write("### Sending Predictions via Telegram")
    try:
        with open(prediction_file, "rb") as file:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendDocument"
            response = requests.post(url, data={"chat_id": CHAT_ID}, files={"document": file})
            if response.status_code == 200:
                st.success("Predicted prices file sent successfully via Telegram!")
            else:
                st.error(f"Failed to send file via Telegram. Error: {response.text}")
    except Exception as e:
        st.error(f"An error occurred while sending the file: {e}")
