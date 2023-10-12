# flight delay prediction software by AI model

import pandas as pd  # for tabular data structure
import numpy as np  # for array computation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt


# ====================================================================================================
# step 1 : Data collection and data preprocessing
# ====================================================================================================
# Generate a synthetic data

# Making the seed in random generator
np.random.seed(0)
n_samples = 1000  # represent the total number of samples to be used

# Generating features: Departure time , Airlines , weather condition which will be used in model trainings

departure_time = np.random.randint(0, 24, n_samples)
# print(departure_time)
airline = np.random.choice(['AirTanzania', 'FlightLink', 'PrecisionAir'], n_samples)
# print(airline)
weather = np.random.choice(['clear', 'Rainy', 'stormy'], size=n_samples)
# print(weather)

# Generating a target variable : Flight delay in minutes ( assuming delays up to 3 hours)
flight_delay = np.random.randint(0, 180, size=n_samples)
# print(flight_delay)


# creating the data frame
flight_data = pd.DataFrame({
    "Airline": airline,
    "DepartureTime": departure_time,
    "Weather": weather,
    "FlightDelay": flight_delay
})

# Display the data set
# pd.set_option('display.max_rows', None)  # Show all rows
# pd.set_option('display.max_columns', None)  # Show all columns

# print(flight_data)

# ===================================================================================================

# Step3 : Feature Engineering
# ===================================================================================================

# creating the feature to represent the departure time of the day

flight_data["TimeOfDay"] = pd.cut(flight_data["DepartureTime"], bins=[0, 6, 12, 18, 24],
                                  labels=["MidNights", "Morning", "Afternoon", "Evening"],
                                  include_lowest=True)
# displaying updated data frame
# print(flight_data)


# =================================================================================================
# Step 4 : Model selection and trainings
# =================================================================================================

# By using simple linear regression model

# select feature (X) and target variable (Y)
# features and target are used to train the modals in Linear  regression

features = ["Airline", "Weather", "DepartureTime"]
target = "FlightDelay"

x = flight_data[features]
y = flight_data[target]

# print(x, y)
# prep the real data for model training and learning (features & target)

# one-hot encode categorical features (k -1)
x = pd.get_dummies(x, columns=['Airline', 'Weather'], drop_first=True)  # there is the big reason for dropping the first
# pd.set_option('display.max_rows', None)  # Show all rows
# pd.set_option('display.max_columns', None)  # Show all columns
# print(x)

# split data into training and testing set
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)
# (x_train, y_train) - this data will be used to train the model
# (x_test, y_test) - this data will be used to test and store(y_test) the delayed time .

# train the modal
model = LinearRegression()
model.fit(x_train, y_train)      # training my model to learn from the data

# Make prediction
y_pred= model.predict(x_test)

# +============================================================================================

# displaying the actual flight delays and predicted flight delayed alongside
prediction_df = pd.DataFrame({'ActualFlightDelay': y_test,
                              'PredictedFlightDelay': y_pred})
print(prediction_df)
# ==============================================================================
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Flight Delay')
plt.ylabel('Predicted Flight Delay')
plt.title('Actual vs. Predicted Flight Delay')
plt.savefig("prediction_df.png")

