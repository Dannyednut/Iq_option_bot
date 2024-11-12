import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from datetime import datetime, timedelta


def NextCandle(data):
    # Step 2: Data Preprocessing
    data['open'] = data['open'].astype(float)
    data['high'] = data['high'].astype(float)
    data['low'] = data['low'].astype(float)
    data['close'] = data['close'].astype(float)

    # Step 3: Feature Engineering
    data['MA_50'] = data['close'].rolling(window=50).mean()
    data['MA_100'] = data['close'].rolling(window=100).mean()
    data['MA_200'] = data['close'].rolling(window=200).mean()
    data['RSI'] = data['close'].rolling(window=14).mean() / data['close'].rolling(window=14).std()
    data['Bollinger_High'] = data['close'].rolling(window=20).mean() + 2*data['close'].rolling(window=20).std()
    data['Bollinger_Low'] = data['close'].rolling(window=20).mean() - 2*data['close'].rolling(window=20).std()
    data['momentum'] = data['close'].pct_change()
    data['volatility'] = data['close'].rolling(window=20).std()

    # Step 4: Feature Selection
    X = data.drop(['close'], axis=1)
    y = data['close']
    X.fillna(0,inplace=True)
    y = y[X.index]
    selector = SelectKBest(f_classif, k=12)
    X_selected = selector.fit_transform(X, y)

    # Step 5: Model Training
    X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

    # Linear Regression
    lr_model = LinearRegression()
    lr_model.fit(X_train, y_train)
    y_pred_lr = lr_model.predict(X_test)
    linear = r2_score(y_test, y_pred_lr)


    # Random Forest Regressor
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    y_pred_rf = rf_model.predict(X_test)
    random = r2_score(y_test, y_pred_rf)

    # Support Vector Regressor
    svr_model = SVR(kernel='rbf', C=1e3, gamma=0.1)
    svr_model.fit(X_train, y_train)
    y_pred_svr = svr_model.predict(X_test)
    vector = r2_score(y_test, y_pred_svr)


    # ... (rest of the code remains the same)

    # Step 6: Prediction and Trade Entry
    next_candle = pd.DataFrame({'time': [pd.to_datetime(datetime.now())]})
    next_candle['open'] = next_candle['time'].apply(lambda x: data['open'].iloc[-1])
    next_candle['high'] = next_candle['time'].apply(lambda x: data['high'].iloc[-1])
    next_candle['low'] = next_candle['time'].apply(lambda x: data['low'].iloc[-1])
    next_candle['volume'] = next_candle['time'].apply(lambda x: data['volume'].iloc[-1])
    next_candle['MA_50'] = next_candle['time'].apply(lambda x: data['MA_50'].iloc[-1])
    next_candle['MA_100'] = next_candle['time'].apply(lambda x: data['MA_100'].iloc[-1])
    next_candle['MA_200'] = next_candle['time'].apply(lambda x: data['MA_200'].iloc[-1])
    next_candle['RSI'] = next_candle['time'].apply(lambda x: data['RSI'].iloc[-1])
    next_candle['Bollinger_High'] = next_candle['time'].apply(lambda x: data['Bollinger_High'].iloc[-1])
    next_candle['Bollinger_Low'] = next_candle['time'].apply(lambda x: data['Bollinger_Low'].iloc[-1])
    next_candle['momentum'] = next_candle['time'].apply(lambda x: data['momentum'].iloc[-1])
    next_candle['volatility'] = next_candle['time'].apply(lambda x: data['volatility'].iloc[-1])
    next_candle = next_candle.drop(['time'],axis=1)
    # Select the best model based on R-squared value
    models = [linear,random,vector]
    
    if max(models) == models[0]:
        best_model =  lr_model
    elif max(models) == models[1]:
        best_model = rf_model
    elif max(models) == models[2]:
        best_model = svr_model
    else:
        best_model = lr_model


    #Prediction parameters
    next_candle.fillna(0,inplace=True)
    next_candle_selected = selector.transform(next_candle)

    # Predict the next 3 consecutive candle prices
    next_candle_pred = best_model.predict(next_candle_selected)

    # Predict the price direction for the next 3 consecutive candles
    price_direction = np.sign(next_candle_pred - data['close'].iloc[-1])


    # Define risk-reward management parameters
    risk_reward_ratio = 2  # 2:1 risk-reward ratio
    take_profit_distance = 30  # 8 pip take profit distance
    stop_loss_distance = take_profit_distance/risk_reward_ratio  # 4 pip stop loss distance

    output = [next_candle_pred,price_direction]
    # Calculate stop loss and take profit prices
    point = 0.01
    if price_direction[0] > 0:  # Long trade
        
        stop_loss = data['close'].iloc[-1] - stop_loss_distance * point
        take_profit = data['close'].iloc[-1] + take_profit_distance * point
    else:  # Short trade
        
        stop_loss = data['close'].iloc[-1] + stop_loss_distance * point
        take_profit = data['close'].iloc[-1] - take_profit_distance * point

    if __name__ == "__main__":
        print('NextCandle:', output[0][0])
        print('PriceDirection:', output[1][0])
        print('Stop Loss:', stop_loss)
        print('Current price: ',data['close'].iloc[-1])
        print('Take Profit:', take_profit)
    else:
        return output
