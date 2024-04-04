import pandas as pd

# Load the historical weather data
historical_data = pd.read_csv('historical_weather_data.csv')

# Print the column names to verify the existence of 'Next_Month_Avg_Temp'
print(historical_data.columns)
