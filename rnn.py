import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout

class CryptoRNNWithPrevPrice:
    def __init__(self, time_step=20, lstm_units=50, dropout_rate=0.2):
        self.time_step = time_step
        self.lstm_units = lstm_units
        self.dropout_rate = dropout_rate
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))

    def build_model(self, input_shape):
        """Build and compile the LSTM model"""
        model = Sequential()
        model.add(LSTM(self.lstm_units, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(self.dropout_rate))
        model.add(LSTM(self.lstm_units, return_sequences=False))
        model.add(Dropout(self.dropout_rate))
        model.add(Dense(1))
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def train_test_coin(self, train_data, test_data, coin_name):
        """Train on training data and test on separate test data for a specific coin"""
        print(f"\n===== Processing {coin_name} =====")

        # Get the closing prices for the coin
        train_prices = train_data['Closing Price'].values
        test_prices = test_data['Closing Price'].values

        # Print data information
        print(f"Training data points for {coin_name}: {len(train_prices)}")
        print(f"Testing data points for {coin_name}: {len(test_prices)}")

        # Check if we have enough data
        if len(train_prices) <= self.time_step + 1:  # +1 because we need at least one target
            print(f"Warning: Not enough training data for {coin_name}.")
            self.time_step = max(5, len(train_prices) - 10)  # Adjust time step
            print(f"Adjusted time_step to: {self.time_step}")

        # Create features with previous day's price
        train_features = []
        for i in range(1, len(train_prices)):
            # Each feature is [current_price, previous_day_price]
            train_features.append([train_prices[i], train_prices[i-1]])

        # Convert to numpy array
        train_features = np.array(train_features)

        # Normalize the features
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_train_features = self.scaler.fit_transform(train_features)

        # Create sequences for LSTM
        X_train, y_train = [], []
        for i in range(len(scaled_train_features) - self.time_step):
            # Input sequence: time_step days of [current_price, previous_price]
            X_train.append(scaled_train_features[i:i+self.time_step])
            # Target: the next day's price after the sequence
            y_train.append(scaled_train_features[i+self.time_step, 0])  # Index 0 is current_price

        X_train, y_train = np.array(X_train), np.array(y_train)

        if len(X_train) == 0:
            print(f"Error: No training sequences created. Reduce time_step (currently {self.time_step}).")
            return None

        # Debug info
        print(f"Training sequences: {len(X_train)}")
        print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

        # Build and train the model
        self.model = self.build_model((X_train.shape[1], X_train.shape[2]))
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            verbose=1
        )

        # Prepare test data
        # We need the last time_step days from training to start prediction
        last_train_prices = train_prices[-self.time_step-1:]  # Get one extra for prev_day

        # Make predictions for test data
        predictions = []

        # Initial input sequence from training data
        input_sequence = []
        for i in range(1, len(last_train_prices)):
            input_sequence.append([last_train_prices[i], last_train_prices[i-1]])

        input_sequence = np.array(input_sequence)
        scaled_input = self.scaler.transform(input_sequence)

        # For each day in test data, make a prediction and update with actual price
        for i in range(len(test_prices)):
            # Reshape input for LSTM [samples, time steps, features]
            current_input = scaled_input[-self.time_step:].reshape(1, self.time_step, 2)

            # Predict next day's price
            predicted_scaled = self.model.predict(current_input)

            # Get the actual price features [current_price, previous_day_price]
            if i == 0:
                prev_price = last_train_prices[-1]  # Last price from training
            else:
                prev_price = test_prices[i-1]  # Previous day's price

            current_price = test_prices[i]  # Current day's price

            # Create feature vector for the actual day's data
            day_features = np.array([[current_price, prev_price]])
            scaled_day_features = self.scaler.transform(day_features)

            # Add the actual day's data to the input sequence
            scaled_input = np.vstack([scaled_input, scaled_day_features])

            # Convert prediction back to original scale
            # Create a dummy row with the predicted value and a placeholder for prev_price
            dummy_row = np.array([[predicted_scaled[0][0], 0]])  # 0 is a placeholder
            unscaled_prediction = self.scaler.inverse_transform(dummy_row)[0][0]

            # Save the prediction
            predictions.append(unscaled_prediction)

        # Calculate metrics
        mae = np.mean(np.abs(np.array(predictions) - test_prices))
        mse = np.mean((np.array(predictions) - test_prices) ** 2)
        rmse = np.sqrt(mse)

        print(f"Prediction Metrics:")
        print(f"MAE: {mae:.2f}")
        print(f"MSE: {mse:.2f}")
        print(f"RMSE: {rmse:.2f}")

        # Calculate returns for trading performance
        if len(predictions) > 1:
            predicted_returns = np.diff(predictions) / predictions[:-1]
            actual_returns = np.diff(test_prices) / test_prices[:-1]

            # Calculate trading metrics
            cumulative_return = (1 + predicted_returns).prod() - 1
            annual_return = cumulative_return / len(test_prices) * 365

            # Handle potential division by zero
            if np.std(predicted_returns) == 0:
                sharpe_ratio = 0
            else:
                sharpe_ratio = np.mean(predicted_returns) / np.std(predicted_returns) * np.sqrt(252)

            # Calculate maximum drawdown
            cumulative_returns_series = (1 + predicted_returns).cumprod()
            cumulative_max = np.maximum.accumulate(cumulative_returns_series)
            drawdowns = cumulative_max - cumulative_returns_series
            max_drawdown = np.max(drawdowns) if len(drawdowns) > 0 else 0

            print(f"Trading Performance:")
            print(f"Cumulative Return: {cumulative_return:.2%}")
            print(f"Annual Return: {annual_return:.2%}")
            print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
            print(f"Max Drawdown: {max_drawdown:.2%}")
        else:
            print("Not enough predictions to calculate trading metrics")
            cumulative_return = annual_return = sharpe_ratio = max_drawdown = 0

        # Return results for this coin
        return {
            'coin': coin_name,
            'predictions': predictions,
            'actual_prices': test_prices.tolist(),
            'dates': test_data['Date'].tolist() if 'Date' in test_data.columns else list(range(len(test_prices))),
            'metrics': {
                'mae': mae,
                'mse': mse,
                'rmse': rmse,
                'cumulative_return': cumulative_return,
                'annual_return': annual_return,
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown
            }
        }

    def process_all_coins(self, train_csv, test_csv):
        """Process all coins using separate train and test CSVs"""
        # Load the datasets
        train_data = pd.read_csv(train_csv)
        test_data = pd.read_csv(test_csv)

        # Get unique coins that exist in both datasets
        train_coins = set(train_data['Coin'].unique())
        test_coins = set(test_data['Coin'].unique())
        common_coins = train_coins.intersection(test_coins)

        print(f"Found {len(common_coins)} common coins in both datasets: {common_coins}")

        results = []

        # Process each coin
        for coin in common_coins:
            train_coin_data = train_data[train_data['Coin'] == coin].reset_index(drop=True)
            test_coin_data = test_data[test_data['Coin'] == coin].reset_index(drop=True)

            result = self.train_test_coin(train_coin_data, test_coin_data, coin)
            if result:
                results.append(result)
                self.plot_predictions(result)

        # Summarize results
        print("\n===== Summary of Results =====")
        for result in results:
            coin = result['coin']
            metrics = result['metrics']
            print(f"{coin}: RMSE={metrics['rmse']:.2f}, Return={metrics['cumulative_return']:.2%}, Sharpe={metrics['sharpe_ratio']:.2f}")

        return results
