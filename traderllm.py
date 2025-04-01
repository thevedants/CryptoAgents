class TraderLLM:
    def __init__(self, api_key=None, model_name="deepseek-ai/DeepSeek-R1"):
        """Initialize the Trader LLM Agent"""
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        self.llm = ChatTogether(
            model=model_name,
            together_api_key=self.api_key
        )
        self.prompt_template = PromptTemplate(
            input_variables=["coin", "analyst_report", "market_context", "portfolio_state", "risk_parameters"],
            template="""
            You are an expert cryptocurrency trader responsible for making final trading decisions.

            COIN: {coin}

            ANALYST REPORT:
            {analyst_report}

            MARKET CONTEXT:
            {market_context}

            CURRENT PORTFOLIO:
            {portfolio_state}

            RISK PARAMETERS:
            {risk_parameters}

            Based on all this information, provide output as per the following

            I only want JSON output in the following format:
            {{{{Decision: BUY, HOLD, SELL}}, {{Coin: Coin Ticker}}, {{Position: Percentage of Portfolio we want to buy/sell}}}}
            Nothing else to be output



            Your decision should consider both the technical analysis and risk management.
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def make_decision(self, coin, analyst_report, market_context, portfolio_state, risk_parameters):
        """Generate trading decision based on analyst report and other factors"""

        decision = self.chain.run({
            "coin": coin,
            "analyst_report": analyst_report,
            "market_context": market_context,
            "portfolio_state": portfolio_state,
            "risk_parameters": risk_parameters
        })

        return decision

def calculate_technical_indicators(prices, volumes=None):
    """Calculate technical indicators from price data"""
    df = pd.DataFrame()
    df['Close'] = prices

    if volumes is not None:
        df['Volume'] = volumes

    # Simple Moving Averages
    df['SMA_5'] = df['Close'].rolling(window=5).mean()
    df['SMA_20'] = df['Close'].rolling(window=20).mean()

    # Exponential Moving Averages
    df['EMA_12'] = df['Close'].ewm(span=12, adjust=False).mean()
    df['EMA_26'] = df['Close'].ewm(span=26, adjust=False).mean()

    # MACD
    df['MACD'] = df['EMA_12'] - df['EMA_26']
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # RSI
    delta = df['Close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # Bollinger Bands
    df['BB_Middle'] = df['Close'].rolling(window=20).mean()
    std_dev = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Middle'] + (std_dev * 2)
    df['BB_Lower'] = df['BB_Middle'] - (std_dev * 2)

    # Stochastic Oscillator
    low_14 = df['Close'].rolling(window=14).min()
    high_14 = df['Close'].rolling(window=14).max()
    df['%K'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['%D'] = df['%K'].rolling(window=3).mean()

    # Zero Mean Reversion (mentioned in the paper)
    df['ZMR'] = (df['Close'] - df['Close'].rolling(window=20).mean()) / (df['Close'].rolling(window=20).std())

    # Drop NaN values
    df = df.dropna()

    # Convert to dictionary for the last row (current indicators)
    indicators = df.iloc[-1].to_dict() if not df.empty else {}

    return indicators

class CryptoAgentsFramework:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        self.rnn_model = CryptoRNNWithPrevPrice(time_step=20)
        self.analyst_llm = TechnicalAnalystLLM(api_key=self.api_key)
        self.trader_llm = TraderLLM(api_key=self.api_key)

    def run_trading_pipeline(self, train_csv, test_csv, market_context=None, portfolio_state=None, risk_parameters=None):
        """Run the complete trading pipeline with RNN and LLMs"""
        # Set default values if not provided
        if market_context is None:
            market_context = f"""
            As of Friday, March 14, 2025, 3:46 PM PDT:
            - Overall crypto market capitalization: $2.1 trillion
            - Bitcoin dominance: 48.5%
            - Recent news: SEC approves new spot ETH ETF applications
            - Market sentiment: Moderately bullish
            - Major events: Bitcoin halving expected in 2 weeks
            """

        if portfolio_state is None:
            portfolio_state = """
            Total portfolio value: $10,000
            Current allocation:
            - BTC: 0%
            - ETH: 0%
            - SOL: 0%
            - XRP: 0%
            - DOGE: 0%
            - Cash: 100%

            """

        if risk_parameters is None:
            risk_parameters = """
            - Risk tolerance: Medium
            - Maximum portfolio allocation to single asset: 50%
            """

        # Run the RNN pipeline
        rnn_results = self.rnn_model.process_all_coins(train_csv, test_csv)

        # Process each coin with the LLMs
        llm_results = []

        for result in rnn_results:
            coin = result['coin']
            print(f"\n===== LLM Analysis for {coin} =====")

            # Calculate technical indicators
            technical_indicators = calculate_technical_indicators(result['actual_prices'])

            # Generate analyst report
            analyst_report = self.analyst_llm.analyze(result, technical_indicators)
            print(f"\nAnalyst Report for {coin}:")
            print(analyst_report)

            # Generate trading decision
            trading_decision = self.trader_llm.make_decision(
                coin=coin,
                analyst_report=analyst_report,
                market_context=market_context,
                portfolio_state=portfolio_state,
                risk_parameters=risk_parameters
            )
            print(f"\nTrading Decision for {coin}:")
            print(trading_decision)

            # Store results
            llm_results.append({
                'coin': coin,
                'analyst_report': analyst_report,
                'trading_decision': trading_decision
            })

        return rnn_results, llm_results

# Set your Together API key
import os
os.environ["TOGETHER_API_KEY"] = YOUR_API_KEY

# Initialize the framework
crypto_agents = CryptoAgentsFramework()

# Run the complete pipeline
rnn_results, llm_results = crypto_agents.run_trading_pipeline(
    train_csv="crypto_prices_90_days.csv",
    test_csv="crypto_prices_30_days.csv"
)

# Save results to CSV
prediction_df = pd.DataFrame()
for result in rnn_results:
    coin_df = pd.DataFrame({
        'Date': result['dates'],
        'Coin': result['coin'],
        'Actual_Price': result['actual_prices'],
        'Predicted_Price': result['predictions'],
        'Difference': np.array(result['predictions']) - np.array(result['actual_prices']),
        'Difference_Pct': ((np.array(result['predictions']) - np.array(result['actual_prices'])) /
                          np.array(result['actual_prices'])) * 100
    })
    prediction_df = pd.concat([prediction_df, coin_df])

prediction_df.to_csv('prediction_results.csv', index=False)

# Save LLM analysis results
llm_df = pd.DataFrame(llm_results)
llm_df.to_csv('llm_analysis_results.csv', index=False)

print("\nAll results saved to CSV files")

import os
import pandas as pd
import numpy as np
from langchain_together import ChatTogether
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import json
import re

class DailyTradingSystem:
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        self.analyst_llm = self._setup_analyst_llm()
        self.trader_llm = self._setup_trader_llm()

    def _setup_analyst_llm(self):
        """Initialize the Technical Analyst LLM Agent"""
        analyst = ChatTogether(
            model="deepseek-ai/DeepSeek-R1",
            together_api_key=self.api_key
        )
        # Simplified prompt template without complex calculations
        prompt_template = PromptTemplate(
            input_variables=["coin", "day", "current_price", "predicted_price", "price_change", "price_change_percent", "technical_indicators"],
            template="""
            You are an expert cryptocurrency technical analyst specializing in daily trading decisions.

            COIN: {coin}
            DAY: {day}

            PRICE DATA:
            - Current Price: ${current_price:.2f}
            - RNN Predicted Price: ${predicted_price:.2f}
            - Predicted Change: ${price_change:.2f}
            - Predicted Change %: {price_change_percent:.2f}%

            TECHNICAL INDICATORS:
            {technical_indicators}

            Provide a concise technical analysis covering:
            1. Interpretation of the RNN model prediction
            2. Analysis of key technical indicators
            3. Overall market trend assessment (bullish, bearish, or neutral)

            Your analysis should be brief but comprehensive enough for the trader to make an informed decision.
            """
        )
        return LLMChain(llm=analyst, prompt=prompt_template)

    def extract_trading_signal(self, llm_output, default_coin="BTC"):
        """
        Extract the trading signal JSON from the LLM output.

        Args:
            llm_output (str): The full text output from the trader LLM
            default_coin (str): Default coin to use if not found in output

        Returns:
            dict: The parsed trading decision with Decision, Coin, and Position
        """
        # Try to find JSON pattern at the end of the output
        json_pattern = r'\{(?:[^{}]|"[^"]*")*\}'
        matches = re.findall(json_pattern, llm_output)

        if matches:
            # Take the last match as it's likely the decision JSON
            try:
                decision = json.loads(matches[-1])
                # Validate that it has the required fields
                if all(key in decision for key in ["Decision", "Coin", "Position"]):
                    return decision
            except json.JSONDecodeError:
                pass

        # If no valid JSON found or parsing failed, extract using string search
        decision_pattern = r'"Decision":\s*"(BUY|SELL|HOLD)"'
        coin_pattern = r'"Coin":\s*"([^"]+)"'
        position_pattern = r'"Position":\s*(\d+)'

        decision_match = re.search(decision_pattern, llm_output)
        coin_match = re.search(coin_pattern, llm_output)
        position_match = re.search(position_pattern, llm_output)

        if decision_match:
            return {
                "Decision": decision_match.group(1),
                "Coin": coin_match.group(1) if coin_match else default_coin,
                "Position": int(position_match.group(1)) if position_match else 0
            }

        # Default fallback if no pattern is found
        return {"Decision": "HOLD", "Coin": default_coin, "Position": 0}


    def _setup_trader_llm(self):
        """Initialize the Trader LLM Agent"""
        trader = ChatTogether(
            model="deepseek-ai/DeepSeek-R1",
            together_api_key=self.api_key
        )
        prompt_template = PromptTemplate(
            input_variables=["coin", "day", "analyst_report", "current_price", "predicted_price", "portfolio_state"],
            template="""
            You are an expert cryptocurrency trader responsible for making daily trading decisions.

            COIN: {coin}
            DAY: {day}

            ANALYST REPORT:
            {analyst_report}

            PRICE DATA:
            - Current Price: ${current_price:.2f}
            - RNN Predicted Price: ${predicted_price:.2f}

            PORTFOLIO STATE:
            {portfolio_state}

            Based on this information, provide your analysis and trading recommendation.

            After your analysis, you MUST end your response with a JSON object in the following format:
            ```
            {{"Decision": "[BUY/SELL/HOLD]", "Coin": "{coin}", "Position": [PERCENTAGE]}}
            ```

            Where:
            - Decision: Must be exactly "BUY", "SELL", or "HOLD"
            - Coin: The cryptocurrency symbol (e.g., "BTC")
            - Position: Recommended position size as a percentage of portfolio (0-100)
            """
        )
        return LLMChain(llm=trader, prompt=prompt_template)


    def calculate_technical_indicators(self, prices_history):
        """Calculate technical indicators from price history"""
        if len(prices_history) < 5:
            return "Insufficient price history for technical indicators"

        # Convert to DataFrame for calculations
        df = pd.DataFrame({'Close': prices_history})

        # Simple Moving Averages
        sma5 = df['Close'].rolling(window=5).mean().iloc[-1] if len(prices_history) >= 5 else None
        sma20 = df['Close'].rolling(window=20).mean().iloc[-1] if len(prices_history) >= 20 else None

        # Exponential Moving Averages
        ema12 = df['Close'].ewm(span=12, adjust=False).mean().iloc[-1] if len(prices_history) >= 12 else None
        ema26 = df['Close'].ewm(span=26, adjust=False).mean().iloc[-1] if len(prices_history) >= 26 else None

        # MACD
        macd = ema12 - ema26 if (ema12 is not None and ema26 is not None) else None

        # RSI
        if len(prices_history) >= 14:
            delta = df['Close'].diff()
            gain = delta.where(delta > 0, 0)
            loss = -delta.where(delta < 0, 0)
            avg_gain = gain.rolling(window=14).mean().iloc[-1]
            avg_loss = loss.rolling(window=14).mean().iloc[-1]

            if avg_loss != 0:
                rs = avg_gain / avg_loss
                rsi = 100 - (100 / (1 + rs))
            else:
                rsi = 70  # Default when no losses (strong uptrend)
        else:
            rsi = None

        # Fixed version with proper f-string formatting
        sma5_str = f"${sma5:.2f}" if sma5 is not None else "N/A"
        sma20_str = f"${sma20:.2f}" if sma20 is not None else "N/A"
        ema12_str = f"${ema12:.2f}" if ema12 is not None else "N/A"
        ema26_str = f"${ema26:.2f}" if ema26 is not None else "N/A"
        macd_str = f"${macd:.2f}" if macd is not None else "N/A"
        rsi_str = f"{rsi:.2f}" if rsi is not None else "N/A"

        indicators = f"""
        SMA(5): {sma5_str}
        SMA(20): {sma20_str}
        EMA(12): {ema12_str}
        EMA(26): {ema26_str}
        MACD: {macd_str}
        RSI: {rsi_str}
        """

        return indicators


def run_daily_trading_simulation(rnn_results, initial_capital=1000):
    """
    Run a daily trading simulation using RNN predictions and LLM decisions

    Args:
        rnn_results: Results from the RNN model containing predictions
        initial_capital: Starting capital for each coin (default: $1000)

    Returns:
        DataFrame with daily trading decisions and portfolio values
    """
    # Set your Together API key
    api_key = YOUR_API_KEY

    # Initialize the trading system
    trading_system = DailyTradingSystem(api_key=api_key)

    # Initialize portfolio for each coin
    portfolios = {}
    all_trading_results = []

    # Process each coin separately
    for result in rnn_results:
        coin = result['coin']
        dates = result['dates']
        actual_prices = result['actual_prices']
        predictions = result['predictions']

        print(f"\n===== Daily Trading Simulation for {coin} =====")

        # Initialize portfolio with $1000
        portfolio = {
            'capital': initial_capital,
            'coin_holdings': 0.0,
            'history': []
        }
        portfolios[coin] = portfolio

        # Keep track of price history for technical indicators
        price_history = []

        # Simulate trading for each day
        for day in range(len(actual_prices)):
            current_price = actual_prices[day]
            predicted_price = predictions[day]
            date = dates[day] if day < len(dates) else f"Day {day}"

            # Add current price to history
            price_history.append(current_price)

            # Calculate technical indicators
            technical_indicators = trading_system.calculate_technical_indicators(price_history)

            # Calculate current portfolio value
            portfolio_value = portfolio['capital'] + portfolio['coin_holdings'] * current_price

            # Format portfolio state
            portfolio_state = f"""
            Current capital: ${portfolio['capital']:.2f}
            Coin holdings: {portfolio['coin_holdings']} {coin}
            Coin value: ${portfolio['coin_holdings'] * current_price:.2f}
            Total portfolio value: ${portfolio_value:.2f}
            """

            # Calculate price differences manually
            price_change = predicted_price - current_price
            price_change_percent = (price_change / current_price) * 100 if current_price > 0 else 0

            # Get analyst report
            analyst_report = trading_system.analyst_llm.run({
                "coin": coin,
                "day": f"Day {day+1} ({date})",
                "current_price": current_price,
                "predicted_price": predicted_price,
                "price_change": price_change,
                "price_change_percent": price_change_percent,
                "technical_indicators": technical_indicators
            })

            # Get trading decision
            full_decision = trading_system.trader_llm.run({
                "coin": coin,
                "day": f"Day {day+1} ({date})",
                "analyst_report": analyst_report,
                "current_price": current_price,
                "predicted_price": predicted_price,
                "portfolio_state": portfolio_state
            })

            # Extract the structured decision from the output
            decision_data = trading_system.extract_trading_signal(full_decision, default_coin=coin)
            decision = decision_data["Decision"]
            position_size = decision_data["Position"]

            # Print the extracted decision
            print(f"Extracted decision: {decision_data}")

            # Execute the trade with position sizing
            previous_portfolio_value = portfolio_value

            if decision == "BUY" and portfolio['capital'] > 0:
                # Use position size to determine how much to invest
                amount_to_invest = portfolio['capital'] * (position_size / 100)
                portfolio['coin_holdings'] += amount_to_invest / current_price
                portfolio['capital'] -= amount_to_invest
            elif decision == "SELL" and portfolio['coin_holdings'] > 0:
                # Use position size to determine how much to sell
                amount_to_sell = portfolio['coin_holdings'] * (position_size / 100)
                portfolio['capital'] += amount_to_sell * current_price
                portfolio['coin_holdings'] -= amount_to_sell

            # Recalculate portfolio value after trade
            portfolio_value = portfolio['capital'] + portfolio['coin_holdings'] * current_price

            # Calculate daily return
            daily_return = (portfolio_value / previous_portfolio_value - 1) * 100 if previous_portfolio_value > 0 else 0

            # Store the result
            trading_result = {
                'Coin': coin,
                'Date': date,
                'Day': day + 1,
                'Current_Price': current_price,
                'Predicted_Price': predicted_price,
                'Decision': decision,
                'Capital': portfolio['capital'],
                'Coin_Holdings': portfolio['coin_holdings'],
                'Portfolio_Value': portfolio_value,
                'Daily_Return': daily_return
            }

            # Add to history
            portfolio['history'].append(trading_result)
            all_trading_results.append(trading_result)

            # Print daily result
            print(f"Day {day+1} ({date}): Price=${current_price:.2f}, Prediction=${predicted_price:.2f}, Decision={decision}, Portfolio=${portfolio_value:.2f}")

    # Convert all results to DataFrame
    df_results = pd.DataFrame(all_trading_results)

    # Save results to CSV
    df_results.to_csv('daily_trading_results.csv', index=False)
    print("\nDaily trading results saved to 'daily_trading_results.csv'")

    # Generate separate CSV files for each coin
    for coin in portfolios:
        coin_results = pd.DataFrame(portfolios[coin]['history'])
        coin_results.to_csv(f'{coin}_trading_results.csv', index=False)
        print(f"{coin} trading results saved to '{coin}_trading_results.csv'")

    return df_results

# Main execution
if __name__ == "__main__":
    # Set your Together API key
    os.environ["TOGETHER_API_KEY"] = YOUR_API_KEY

    # Import the CryptoAgentsFramework
    # Initialize the framework
    crypto_agents = CryptoAgentsFramework()

    # Run the RNN pipeline to get predictions
    rnn_results, _ = crypto_agents.run_trading_pipeline(
        train_csv="crypto_prices_90_days.csv",
        test_csv="crypto_prices_30_days.csv"
    )

    # Run daily trading simulation using RNN predictions with $1000 initial capital
    trading_results = run_daily_trading_simulation(rnn_results, initial_capital=1000)

    # Calculate final performance metrics for each coin
    print("\n===== Trading Performance Summary =====")
    for coin in trading_results['Coin'].unique():
        coin_results = trading_results[trading_results['Coin'] == coin]
        initial_value = 1000  # $1000 per coin
        final_value = coin_results['Portfolio_Value'].iloc[-1]
        total_return = (final_value / initial_value - 1) * 100

        # Calculate Sharpe ratio
        daily_returns = coin_results['Daily_Return'].values
        sharpe_ratio = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0

        # Calculate max drawdown
        portfolio_values = coin_results['Portfolio_Value'].values
        peak = np.maximum.accumulate(portfolio_values)
        drawdown = (peak - portfolio_values) / peak
        max_drawdown = np.max(drawdown) * 100

        print(f"{coin}:")
        print(f"  Initial Value: ${initial_value:.2f}")
        print(f"  Final Value: ${final_value:.2f}")
        print(f"  Total Return: {total_return:.2f}%")
        print(f"  Sharpe Ratio: {sharpe_ratio:.2f}")
        print(f"  Max Drawdown: {max_drawdown:.2f}%")
