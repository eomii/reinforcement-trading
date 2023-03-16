import ccxt
import numpy as np
import time
import random
import tensorflow as tf

api_key = ''
secret_key = ''

binance = ccxt.binance({
    'apiKey': api_key,
    'secret': secret_key,
})

symbol = 'SOL/USDT'
timeframe = '1m'
transaction_fee = 0.001
max_risk = 0.05

class MarketMakingStrategy:
    def __init__(self, num_states, num_actions, alpha, gamma, epsilon, model):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.model = model
        self.placed_orders = []

    def get_state(self, data):
        order_book, ohlcv = data
        bids, asks = order_book['bids'], order_book['asks']
        state = np.concatenate((np.array(bids[:6]).flatten(), np.array(asks[:6]).flatten(), ohlcv))
        return state

    def choose_action(self, state):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        else:
            q_values = self.model(state[np.newaxis])
            return np.argmax(q_values.numpy())

    def update_q_network(self, state, action, reward, next_state):
        next_q_values = self.model(next_state[np.newaxis]).numpy()
        target = reward + self.gamma * np.max(next_q_values)
        target_q_values = self.model(state[np.newaxis]).numpy()
        target_q_values[0, action] = target
        
        self.model.fit(state[np.newaxis], target_q_values, epochs=1, verbose=0)

    def get_balance(self, currency):
        balances = binance.fetch_balance()
        for balance in balances['info']['balances']:
            if balance['asset'] == currency:
                return float(balance['free'])
        return 0

    def execute_action(self, action):
        order_book = binance.fetch_order_book(symbol)
        bids, asks = order_book['bids'], order_book['asks']

        binance.load_markets()
        market_info = binance.market(symbol)
        min_trade_amount = market_info['limits']['amount']['min']

        symbol_info = [s for s in binance.markets.values() if s['symbol'] == symbol][0]
        min_notional = float([f['minNotional'] for f in symbol_info['info']['filters'] if f['filterType'] == 'MIN_NOTIONAL'][0])

        min_usd_amount = 6
        buffer = 1.05  # 1% buffer

        if action == 0:  # Buy
            usdt_balance = self.get_balance('USDT')
            if usdt_balance >= min_notional:
                price = asks[0][0] * (1 - transaction_fee)
                amount = max(max_risk / price, min_trade_amount)
                amount = max(amount, max(min_notional * buffer / price, min_usd_amount * buffer / price))
                amount = round(amount, 5)  # Round amount after updating

                print("Price:", price)
                print("Amount:", amount)
                print("Notional:", price * amount)
                print("Minimum notional:", min_notional)

                try:
                    order = binance.create_limit_buy_order(symbol, amount, price)
                    self.placed_orders.append((time.time(), order))
                except Exception as e:
                    print("Error creating buy order:", e)

        elif action == 1:  # Sell
            btc_balance = self.get_balance('SOL')
            if btc_balance * bids[0][0] >= min_notional:
                price = bids[0][0] * (1 + transaction_fee)
                amount = max(max_risk / price, min_trade_amount)
                amount = max(amount, max(min_notional * buffer / price, min_usd_amount * buffer / price))
                amount = round(amount, 5)  # Round amount after updating

                print("Price:", price)
                print("Amount:", amount)
                print("Notional:", price * amount)
                print("Minimum notional:", min_notional)

                try:
                    order = binance.create_limit_sell_order(symbol, amount, price)
                    self.placed_orders.append((time.time(), order))
                except Exception as e:
                    print("Error creating sell order:", e)

        else:  # Hold
            pass


    def cancel_old_orders(self, max_age_seconds=15):
        current_time = time.time()
        orders_to_remove = []

        for i, (order_time, order) in enumerate(self.placed_orders):
            if current_time - order_time > max_age_seconds:
                try:
                    binance.cancel_order(symbol, order['id'])
                    print(f"Order {order['id']} canceled.")
                    orders_to_remove.append(i)
                except Exception as e:
                    print(f"Error canceling order {order['id']}:", e)

        for index in sorted(orders_to_remove, reverse=True):
            del self.placed_orders[index]


    def get_reward(self, action):
        order_book = binance.fetch_order_book(symbol)
        bids, asks = order_book['bids'], order_book['asks']
        current_mid_price = (bids[0][0] + asks[0][0]) / 2

        self.execute_action(action)
        time.sleep(5)

        new_order_book = binance.fetch_order_book(symbol)
        new_bids, new_asks = new_order_book['bids'], new_order_book['asks']
        new_mid_price = (new_bids[0][0] + new_asks[0][0]) / 2

        mid_price_difference = new_mid_price - current_mid_price

        if action == 0:  # Buy
            reward = mid_price_difference
        elif action == 1:  # Sell
            reward = -mid_price_difference
        else:  # Hold
            reward = 0

        return reward

    def run(self):
        while True:
            data = (binance.fetch_order_book(symbol), binance.fetch_ohlcv(symbol, timeframe)[-1])
            print("data")
            state = self.get_state(data)
            print("state")
            action = self.choose_action(state)
            print("action")
            reward = self.get_reward(action)
            print("reward")
            next_data = (binance.fetch_order_book(symbol), binance.fetch_ohlcv(symbol, timeframe)[-1])
            print("next_data")
            next_state = self.get_state(next_data)
            print("next_state")

            self.update_q_network(state, action, reward, next_state)
            self.cancel_old_orders()

            time.sleep(5)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(30,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3)
])

model.compile(optimizer='adam', loss='mse')

market_maker = MarketMakingStrategy(30, 3, 0.1, 0.99, 0.1, model)
market_maker.run()
