import time
from datetime import datetime, timedelta
import pytz # install this library using pip

# Set the timezone to Eastern Standard Time (EST)
tz = pytz.timezone('US/Eastern')

# Define the time for buying and selling
buy_time = datetime.now(tz).replace(hour=19, minute=45, second=0, microsecond=0)
sell_time = buy_time + timedelta(days=1, hours=8, minutes=30)

# Define the buy and sell functions
def buy_QQQ():
    # Place buy order for QQQ
    print("Buy order placed at", buy_time)

def sell_QQQ():
    # Place sell order for QQQ
    print("Sell order placed at", sell_time)

# Check the time every 30 seconds and execute the buy and sell functions if the time is right
while True:
    current_time = datetime.now(tz)
    if current_time >= buy_time and current_time < sell_time:
        buy_QQQ()
    elif current_time >= sell_time:
        sell_QQQ()
        break
    time.sleep(30)