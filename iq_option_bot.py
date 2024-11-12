# bot.py
import asyncio
from iqoptionapi.stable_api import IQ_Option
from talib import *
import numpy as np
import pandas as pd 
from telegram import Bot, Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters
from datetime import datetime
import time
import logging
from model import NextCandle
from config_manager import ConfigManager

class IQ_options:
    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        credentials = self.config.get_credentials()
        
        self.email = credentials['email']
        self.password = credentials['password']
        self.telegram_token = credentials['telegram_token']
        self.channel_ids = [self.config.get_channels()[0]]
        
        trading_settings = self.config.get_trading_settings()
        self.active_account_type = trading_settings['account_type']
        
        self.logging = logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
        self.api = IQ_Option(self.email, self.password)
        self.application = Application.builder().token(self.telegram_token).build()
        self.application.add_handler(CommandHandler("start", self.start_command))
        self.application.add_handler(CommandHandler("trade", self.trade_command))
        self.application.add_handler(MessageHandler(
            filters.ChatType.CHANNEL & filters.COMMAND,
            self.handle_channel_command
        ))        
        self.connect = self.api.connect()

    async def start_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle start command in private chats and groups"""
        if update.message:
            await update.message.reply_text('Bot started! Use /trade to start trading.')

    async def trade_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle trade command in private chats and groups"""
        if update.message:
            symbol = self.config.get_trading_settings()['default_symbol']
            await self.execute_trade(symbol)

    async def handle_channel_command(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Handle commands posted in channels"""
        if not update.channel_post:
            print('nOT CHANNEL')
            return

        command = update.channel_post.text.split()[0].lower()
        channel_username = update.channel_post.chat.id
        print(channel_username)
        if f"{channel_username}" in self.channel_ids:
            if command == '/start':
                await self.send_telegram_message('Bot started in channel! Use /trade to start trading.')
            elif command == '/trade':
                symbol = self.config.get_trading_settings()['default_symbol']
                await self.execute_trade(symbol)
        else:
            logging.error(f'Not match with {channel_username}')

    async def send_telegram_message(self, message):
        try:
            for channel_id in self.channel_ids:
                print(channel_id)
                await self.application.bot.send_message(chat_id=channel_id, text=message, parse_mode='HTML')
            logging.info(f"Telegram message sent: {message}")
        except Exception as e:
            logging.error(f"Failed to send Telegram message: {e}")
            quit()

    def get_candle_data(self, symbol: str, bars: int) -> pd.DataFrame:
        self.api.start_candles_stream(symbol,60,bars)
        candles = self.api.get_realtime_candles(symbol,60)
        data = []
        for candle in candles:
            # Assuming each candle is a dictionary with the required keys
            data.append({
                'open': candles[candle]['open'],
                'high': candles[candle]['max'],
                'low': candles[candle]['min'],
                'close': candles[candle]['close'],
                'volume': candles[candle]['volume']
            })
        iq_bot.api.stop_candles_stream(symbol,60)   
        inputs = pd.DataFrame(data)
        return inputs
    
    def get_bollinger_bands(self, data, period=20):
        close_prices = np.array(data['close'])
        upper_band, mid_band,lower_band = BBANDS(close_prices, timeperiod=period) # type: ignore
        return upper_band, lower_band

    def rsi(self,data,period=14):
        close_prices = np.array(data['close'])
        indicate = RSI(close_prices, timeperiod=period) # type: ignore
        return indicate
    
    def get_price(self,symbol: str):
        data = self.get_candle_data(symbol,1)
        return data['close'].iloc[-1]
    
    def get_ema(self, data, period=100):
        close_prices = np.array(data['close'])
        ema = EMA(close_prices,timeperiod=period)  # type: ignore
        return ema

    def place_trade(self, amount, symbol, direction,expiration):
        result = self.api.buy(amount, symbol, direction, expiration)  # Adjust expiration as needed
        return result

    def calculate_sl_tp(self, entry_price, sl_percent, tp_percent):
        sl_price = entry_price * (1 - sl_percent / 100)
        tp_price = entry_price * (1 + tp_percent / 100)
        return sl_price, tp_price
    
    async def execute_trade(self, symbol, expiration=1):
        risk_settings = self.config.get_risk_settings()
        gale_settings = self.config.get_gale_settings()
        indicator_settings = self.config.get_indicator_settings()
        message_templates = self.config.get_message_templates()

        data = self.get_candle_data(symbol, 100)
        '''
        upper_band, lower_band = self.get_bollinger_bands(
            data, 
            period=indicator_settings['bollinger_bands']['period']
        )
        ema = self.get_ema(
            data, 
            period=indicator_settings['ema']['period']
        )
        rsi_values = self.rsi(
            data,
            period=indicator_settings['rsi']['period']
        )
        '''
        direction = NextCandle(data)[1][0]
        print(direction)
        amount = self.config.get_trading_settings()['amount']
        gale_count = 0

        while gale_count < gale_settings['max_gale']:
            # ... [trading logic remains similar, but uses configuration values] ...
            current_price = self.get_price(symbol)
            
            '''
            current_upper_band = upper_band[-1]
            current_lower_band = lower_band[-1]
            current_ema = ema[-1]
            current_rsi = rsi_values[-1]
            '''
            tp_percent = risk_settings['take_profit_percent']
            sl_percent = risk_settings['stop_loss_percent']
            #if (current_price < current_lower_band and current_price > current_ema and current_rsi < 30):
            if direction > 0:
                entry_price = current_price

                sl_price, tp_price = self.calculate_sl_tp(entry_price, sl_percent, tp_percent)
                check, id = self.place_trade(amount, symbol, "call",expiration)
                if check:
                    # Format the message using the template
                    trade_message = message_templates['trade_open'].format(
                        symbol = symbol,
                        direction="CALL",
                        entry_price=entry_price,
                        amount=amount,
                        expiration=expiration,
                        gale=gale_count 
                    )
                    await self.send_telegram_message(trade_message)
                    result,profit = self.api.check_win_v4(id)
                    if result == 'win':
                        trade_message = message_templates['trade_close'].format(
                        symbol = symbol,
                        direction="CALL",
                        entry_price=entry_price,
                        amount=amount,
                        expiration=expiration,
                        gale=gale_count,
                        result=f"✅✅{result.upper()}✅✅" 
                    )
                        await self.send_telegram_message(trade_message)
                        print(True)
                        exit()
                    elif result == 'loose':
                        trade_message = message_templates['trade_close'].format(
                        symbol = symbol,
                        direction="CALL",
                        entry_price=entry_price,
                        amount=amount,
                        expiration=expiration,
                        gale=gale_count,
                        result=f"🤦‍♂️🤦‍♀️{result.upper()}🤦‍♀️🤦‍♂️" 
                    )
                        await self.send_telegram_message(trade_message)
                        gale_count, amount = await self.martingale(gale_count,amount,result)
                    
                else:
                    trade_message = message_templates['trade_error'].format(
                        id=id
                    )
                    await self.send_telegram_message(trade_message)
                    break
            #elif (current_price > current_upper_band and current_price < current_ema and current_rsi > 70):
            elif direction < 0:
                entry_price = current_price
                sl_price, tp_price = self.calculate_sl_tp(entry_price, sl_percent, tp_percent)
                check, id = self.place_trade(amount, symbol, "put", expiration)
                if check:
                    trade_message = message_templates['trade_open'].format(
                        symbol = symbol,
                        direction="PUT",
                        entry_price=entry_price,
                        amount=amount,
                        expiration=expiration,
                        gale=gale_count   
                    )
                    await self.send_telegram_message(trade_message)
                    result,profit = self.api.check_win_v4(id)
                    if result == 'win':
                        trade_message = message_templates['trade_close'].format(
                        symbol = symbol,
                        direction="PUT",
                        entry_price=entry_price,
                        amount=amount,
                        expiration=expiration,
                        gale=gale_count,
                        result=f"✅✅{result.upper()}✅✅" 
                    )
                        await self.send_telegram_message(trade_message)
                        print(True)
                        exit()
                    elif result == 'loose':
                        trade_message = message_templates['trade_close'].format(
                        symbol = symbol,
                        direction="PUT",
                        entry_price=entry_price,
                        amount=amount,
                        expiration=expiration,
                        gale=gale_count,
                        result=f"🤦‍♂️🤦‍♀️{result.upper()}🤦‍♀️🤦‍♂️" 
                    )
                        await self.send_telegram_message(trade_message)
                        gale_count, amount = await self.martingale(gale_count,amount,result)
                    
                else:
                    trade_message = message_templates['trade_error'].format(
                        id=id
                    )
                    await self.send_telegram_message(trade_message)
                    break
        return


    async def martingale(self, gale_count: int, amount: int, condition: str):
        if condition == 'loose':
            gale_count += 1
            amount *= 2  # Gale strategy: double the amount for the next trade
            await self.send_telegram_message(f"Martingale increase: {gale_count}")
            await asyncio.sleep(60)  # Wait for the next candle
            return (gale_count,amount)
        elif condition == 'win' and gale_count>1:
            gale_count -= 1
            amount /= 2  # Gale strategy: double the amount for the next trade
            await self.send_telegram_message(f"Martingale increase: {gale_count}")
            await asyncio.sleep(60)  # Wait for the next candle
            return (gale_count,amount)
        elif condition == 'win':
            return (gale_count,amount)

                
    def run(self):
        try:
            """Run the bot"""
            
            self.application.run_polling()
   
        except Exception as e:
            logging.error(f"Error in running bot: {e}")
            self.application.stop()

    async def start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        await update.message.reply_text('Bot started!')

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Create bot instance with configuration
    iq_bot = IQ_options("config.yaml")

    try:
        check, reason = iq_bot.connect
        if check:
            iq_bot.run()
    except KeyboardInterrupt:
        logging.info("Bot stopped by user.")
    except Exception as e:
        logging.error(f"An error occurred: {e}")
        