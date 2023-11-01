import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import StandardScaler
df = pd.read_csv("Electricity.csv", na_values=['?'])
df.isnull().sum()
df = df.dropna()

#Splitting the independent features and target feature
X = df[['ActualWindProduction', 'SystemLoadEP2', 'SMPEA', 'SystemLoadEA', 'ForecastWindProduction', 
     'DayOfWeek', 'Year', 'ORKWindspeed', 'CO2Intensity', 'PeriodOfDay']]
y = df['SMPEP2']
#Train-Validation Split (90% Train set and 10% Validation set)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 42)
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
model = keras.Sequential([
        keras.layers.Dense(512, activation="relu", input_shape=[10]),
        keras.layers.Dense(800, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1024, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation = 'linear'),
        ])
model.summary()
model.compile(loss='mse', optimizer='adam', metrics=['mse','mae'])
#Fitting the model with Early stopping and restoring the best weights
early_stopping = keras.callbacks.EarlyStopping(patience = 10, min_delta = 0.001, restore_best_weights =True )
history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=50,
    epochs=100,
    callbacks=[early_stopping],
    verbose=1, 
)
#Evaluating the model on test set
from sklearn.metrics import mean_absolute_error,r2_score
predictions = model.predict(X_test)
print(f"MAE: {mean_absolute_error(y_test, predictions)}")

print(f"R2_score: {r2_score(y_test, predictions)}")
#XG Boost Regressor Model
from xgboost import XGBRegressor
model2 = XGBRegressor(n_estimators = 8000, max_depth=17, eta=0.1, subsample=0.7, colsample_bytree=0.8)
model2.fit(X_train, y_train)
pred = model2.predict(X_test)
r2_score(y_test, pred)

mean_absolute_error(y_test, pred)




