# Market Making Trading Bot

This trading bot is designed to perform market making strategies on the Binance exchange for the SOL/USDT trading pair. It uses reinforcement learning with a Q deep neural network to make trading decisions. The bot interacts with the Binance API using the CCXT library, and it operates on a 5-second timeframe.

## Features

- Market making strategy based on reinforcement learning
- Q deep neural network for decision making
- Order management with risk control
- Fetches live data using Binance WebSocket API
- Ensures minimum notional value and complies with Binance trading rules

## Requirements

- Python 3.6 or higher
- CCXT library
- NumPy
- TensorFlow

## Installation

1. Clone this repository:

```bash
git clone https://github.com/eomii/reinforcement-trading.git
```

2. Install the required libraries:

```bash
pip install -r requirements.txt
```


3. Set up your Binance API key and secret in the `qdnnv1.py` file.

```python
API_KEY = 'your-api-key'
API_SECRET = 'your-api-secret'
```

## Usage

1. Run the trading bot:

```python
python qdnnv1.py
```


2. Monitor the bot's trading activity and performance.

**Note:** This trading bot is for educational purposes only. Use it at your own risk. The developers are not responsible for any financial losses incurred while using this bot.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.

## Contributing

Contributions are welcome! Feel free to open issues or submit pull requests to improve the trading bot.
