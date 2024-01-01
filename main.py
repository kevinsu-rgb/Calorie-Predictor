import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def convert_to_seconds(time_str):
    time_components = list(map(int, time_str.split(':')))
    if len(time_components) == 2:  # MM:SS format
        return time_components[0] * 60 + time_components[1]
    elif len(time_components) == 3:  # HH:MM:SS format
        return time_components[0] * 3600 + time_components[1] * 60 + time_components[2]

if __name__ == '__main__':
    cardio_activities_filepath = 'cardioActivities.csv' # File Name
    cardio_data = pd.read_csv(cardio_activities_filepath)
    df = cardio_data.copy() # Create a copy of CSV file data
    df['Duration'] = df['Duration'].apply(convert_to_seconds) # Convert all the 'Duration' times to seconds
    y = cardio_data.Calories_Burned # Predicting for the calories burned
    cardio_data_features = ['Type', 'Distance_(km)', 'Duration', 'Climb_(m)', 'Average_Heart_Rate_(bpm)'] # Features to be used for the ML model
    X = df[cardio_data_features]
    combined_data = pd.concat([X, y], axis=1) # Combine X and Y
    combined_data = combined_data.dropna(axis=0) # Drop the N/A from both datasets
    X = combined_data.drop('Calories_Burned', axis=1) # Split dataset again to X and y
    y = combined_data['Calories_Burned']
    X = pd.get_dummies(X, columns=['Type'], drop_first=True)
    cardio_model = DecisionTreeRegressor(random_state=1)
    cardio_model.fit(X, y) # Use DecisionTreeRegressor to create a model

    print("Making predictions for the following 5 data:")
    print(X.head()) # Grab the first 5 data points from the dataset
    print("The predictions are")
    print(cardio_model.predict(X.head())) # Prediction
    predicted_calories = cardio_model.predict(X)
    print(f'The mean absoute error is: {mean_absolute_error(y, predicted_calories)}') # Accuracy
    print(f'The mean squared error is: {mean_squared_error(y, predicted_calories)}')
    print(f'Accuracy score (Out of 1.0): {r2_score(y, predicted_calories)}')
    print("")

    all_possible_types = ['Running', 'Walking',' Cycling', 'Other']  # Add all possible categories for 'Type'
    user_input = {
        'Type': input("Enter the activity type ((Running, Walking, Cycling, Other)): ") , # User inputs
        'Distance_(km)': float(input("Enter the distance in kilometers: ")),
        'Duration': convert_to_seconds(input("Enter the duration in seconds (Minutes:Seconds or Hours:Minutes:Seconds): ")),
        'Climb_(m)': float(input("Enter the climb in meters: ")),
        'Average_Heart_Rate_(bpm)': float(input("Enter the average heart rate in bpm: "))
    }

    user_input_df = pd.DataFrame([user_input])
    user_input_df['Type'] = user_input_df['Type'].apply(lambda x: x if x in all_possible_types else 'Other')
    user_input_df = pd.get_dummies(user_input_df, columns=['Type'], drop_first=True)
    user_input_df = user_input_df.reindex(columns=X.columns, fill_value=0)
    user_predictions = cardio_model.predict(user_input_df)
    print("Predicted Calories_Burned based on user input:")
    print(user_predictions[0]) # Calorie Prediction for user










