import ccxt
import numpy as np
import threading
import time
import random
import tensorflow as tf

api_key = ''
secret_key = ''

binance = ccxt.binance({
    'apiKey': api_key,
    'secret': secret_key,
})

symbol = 'SOL/BUSD'
timeframe = '1m'
transaction_fee = 0.001
max_risk = 0.05

class MarketMakingStrategy:
    def __init__(self, num_states, num_actions, alpha, gamma, epsilon, model):
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.total_balance = 0
        self.epsilon = epsilon
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995       
        self.model = model
        self.placed_orders = []
        self.starting_balance = self.get_total_balance()

    def update_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


    def update_total_balance(self):
        while True:
            self.total_balance = self.get_total_balance()
            print("Total balance is: ", self.total_balance)
            time.sleep(10)

    def start_balance_updater(self):
        balance_updater_thread = threading.Thread(target=self.update_total_balance)
        balance_updater_thread.daemon = True
        balance_updater_thread.start()

    def get_total_balance(self, base_currency='USDT'):
        balances = binance.fetch_balance()
        total_balance = 0
        
        # Fetch all ticker prices in a single API call
        tickers = binance.fetch_tickers()
        available_symbols = set(tickers.keys())

        for balance in balances['info']['balances']:
            asset = balance['asset']
            free_balance = float(balance['free'])
            locked_balance = float(balance['locked'])
            total_asset_balance = free_balance + locked_balance

            if asset == base_currency:
                total_balance += total_asset_balance
            else:
                try:
                    ticker = f'{asset}/{base_currency}'
                    if ticker in available_symbols:
                        asset_price = tickers[ticker]['last']
                        total_balance += total_asset_balance * asset_price
                except ccxt.BaseError as e:
                    print(f"Error fetching ticker for {ticker}: {e}")
        return total_balance

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
            usdt_balance = self.get_balance('BUSD')
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

    def cancel_old_orders(self, max_age_seconds=180):
        current_time = time.time()
        orders_to_remove = []

        for i, (order_time, order) in enumerate(self.placed_orders):
            if current_time - order_time > max_age_seconds:
                try:
                    binance.cancel_order(order['id'], symbol)
                    print(f"Order {order['id']} canceled.")
                    orders_to_remove.append(i)
                except ccxt.OrderNotFound as e:
                    print(f"Order {order['id']} not found or already canceled/filled.")
                    orders_to_remove.append(i)
                except Exception as e:
                    print(f"Error canceling order {order['id']}:", e)

        for index in sorted(orders_to_remove, reverse=True):
            del self.placed_orders[index]

    def update_order_statuses(self):
        updated_orders = []
        for _, order in self.placed_orders:
            try:
                order_info = binance.fetch_order(order['id'], symbol)
                updated_orders.append((_, order_info))
            except Exception as e:
                print(f"Error fetching order {order['id']} status:", e)
        self.placed_orders = updated_orders

    def get_reward(self, action):
        starting_balance = self.starting_balance
        self.execute_action(action)
        time.sleep(5)
        new_balance = self.total_balance

        reward = new_balance - starting_balance

        return reward

    def update_order_statuses_and_remove_filled(self):
        orders_to_remove = []

        for i, (_, order) in enumerate(self.placed_orders):
            try:
                updated_order = binance.fetch_order(order['id'], symbol)
                if updated_order['status'] == 'closed' or updated_order['status'] == 'canceled':
                    orders_to_remove.append(i)
            except Exception as e:
                print(f"Error fetching order {order['id']} status:", e)

        for index in sorted(orders_to_remove, reverse=True):
            del self.placed_orders[index]

    def run(self):
        self.start_balance_updater()
        while True:
            data = (binance.fetch_order_book(symbol), binance.fetch_ohlcv(symbol, timeframe)[-1])
            print("data")
            state = self.get_state(data)
            print("state")
            action = self.choose_action(state)
            print("action")
            self.update_order_statuses()
            reward = self.get_reward(action)
            self.update_order_statuses_and_remove_filled()
            print("reward")
            next_data = (binance.fetch_order_book(symbol), binance.fetch_ohlcv(symbol, timeframe)[-1])
            print("next_data")
            next_state = self.get_state(next_data)
            print("next_state")

            self.update_q_network(state, action, reward, next_state)
            self.update_epsilon() 
            print("epsilon: ", self.epsilon)
            self.cancel_old_orders()

            time.sleep(5)

model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(30,)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(3)
])

model.compile(optimizer='adam', loss='mse')

market_maker = MarketMakingStrategy(30, 3, 0.1, 0.99, 0.99, model)
market_maker.run()
