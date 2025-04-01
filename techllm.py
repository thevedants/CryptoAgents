from langchain_together import ChatTogether
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
import json

class TechnicalAnalystLLM:
    def __init__(self, api_key=None, model_name="deepseek-ai/DeepSeek-R1"):
        """Initialize the Technical Analyst LLM Agent"""
        self.api_key = api_key or os.environ.get("TOGETHER_API_KEY")
        self.llm = ChatTogether(
            model=model_name,
            together_api_key=self.api_key
        )
        self.prompt_template = PromptTemplate(
            input_variables=["coin", "price_data", "rnn_predictions", "technical_indicators", "prediction_metrics"],
            template="""
            You are an expert cryptocurrency technical analyst specializing in pattern recognition and market analysis.

            COIN: {coin}

            PRICE DATA:
            {price_data}

            RNN MODEL PREDICTIONS:
            {rnn_predictions}

            TECHNICAL INDICATORS:
            {technical_indicators}

            RNN MODEL PERFORMANCE METRICS:
            {prediction_metrics}

            Provide a detailed technical analysis covering:
            1. Pattern identification (support/resistance levels, chart patterns)
            2. Interpretation of the RNN model predictions and their reliability
            3. Analysis of technical indicators (MACD, RSI, Bollinger Bands, etc.)
            4. Key levels to watch and potential price targets
            5. Overall market trend assessment (bullish, bearish, or neutral)

            Format your analysis in a clear, structured manner that can be easily interpreted by the trader LLM.
            """
        )
        self.chain = LLMChain(llm=self.llm, prompt=self.prompt_template)

    def analyze(self, result, technical_indicators):
        """Generate technical analysis based on RNN predictions and indicators"""
        coin = result['coin']
        actual_prices = result['actual_prices']
        predictions = result['predictions']
        metrics = result['metrics']

        # Format price data
        price_data = {
            "current_price": float(actual_prices[-1]),
            "price_change_1d": float(actual_prices[-1] - actual_prices[-2]) if len(actual_prices) > 1 else 0,
            "price_change_pct_1d": float((actual_prices[-1] / actual_prices[-2] - 1) * 100) if len(actual_prices) > 1 else 0,
            "price_change_7d": float(actual_prices[-1] - actual_prices[-8]) if len(actual_prices) >= 8 else None,
            "price_change_pct_7d": float((actual_prices[-1] / actual_prices[-8] - 1) * 100) if len(actual_prices) >= 8 else None,
            "30d_high": float(max(actual_prices)),
            "30d_low": float(min(actual_prices))
        }

        # Format RNN predictions
        rnn_data = {
            "latest_prediction": float(predictions[-1]),
            "predicted_change": float(predictions[-1] - actual_prices[-1]),
            "predicted_change_pct": float((predictions[-1] / actual_prices[-1] - 1) * 100),
            "prediction_trend": "Upward" if predictions[-1] > actual_prices[-1] else "Downward",
            "recent_predictions": [float(p) for p in predictions[-5:]]  # Last 5 predictions
        }

        # Run the analysis
        analysis = self.chain.run({
            "coin": coin,
            "price_data": json.dumps(price_data, indent=2),
            "rnn_predictions": json.dumps(rnn_data, indent=2),
            "technical_indicators": json.dumps(technical_indicators, indent=2),
            "prediction_metrics": json.dumps(metrics, indent=2)
        })

        return analysis