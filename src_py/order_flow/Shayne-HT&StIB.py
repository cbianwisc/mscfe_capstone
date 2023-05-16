from ib_insync import *

# Connect to Interactive Brokers
util.startLoop()
ib = IB()
ib.connect('host', port, clientId)

# Define contract details
contract = Stock('QQQ', 'SMART', 'USD')

# Request historical time and sales data
ticks = ib.reqHistoricalTicks(contract, '', '', 1000, 'TRADES', useRth=False)

# Print time and sell price for each tick
for tick in ticks:
    print(f'Time: {tick.time}, Sell Price: {tick.price}')

# Disconnect from Interactive Brokers
ib.disconnect()
