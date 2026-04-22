import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.ensemble import RandomForestRegressor 

# load dataset ( make sure file name matches)
data=pd.read_csv('random_stock_market_dataset.csv')

# sort by date 
data=data.sort_values('Date')

#create moving average 
data['MA_10']= data ['Close'].rolling(window=10).mean()

# drop missing values
data=data.dropna()

# features and target
features=['Open','High','Low','Close','Volume','MA_10']

#predict next day close price
data['Prediction']=data['Close'].shift(-1)
X=np.array(data[features])
y=np.array(data['Prediction'])

# Remove last row because of NaN in prediction 
X=X[:-1]
y=y[:-1]

#Train Model
model= RandomForestRegressor()
model.fit(X,y)

#Predict next day
last_row=data[features].iloc[-1].values.reshape(1,-1)
prediction = model.predict(last_row)

print("Predicted Next Day Close Price", prediction[0])

# Plot graph
plt.plot(data['Close'], label = 'Actual Price')
plt.title("Stock Price Trend")
plt.legend()
plt.show()

