from datetime import datetime
import pytz
from ibapi.client import EClient
from ibapi.wrapper import EWrapper
from ibapi.contract import Contract
from ibapi.order import *

class IBapi(EWrapper, EClient):
    def __init__(self):
        EClient.__init__(self, self)

    def nextOrderId(self):
        oid = self.nextValidId
        self.nextValidId += 1
        return oid

    def orderStatus(self, orderId, status, filled, remaining, avgFillPrice, permId, parentId, lastFillPrice, clientId, whyHeld, mktCapPrice):
        print('orderStatus - orderId:', orderId, 'status:', status, 'filled:', filled, 'remaining:', remaining, 'lastFillPrice:', lastFillPrice)

    def openOrder(self, orderId, contract, order, orderState):
        print('openOrder - orderId:', orderId, contract.symbol, contract.secType, '@', contract.exchange, ':', order.action, order.orderType, order.totalQuantity, orderState.status)

    def execDetails(self, reqId, contract, execution):
        print('execDetails - reqId:', reqId, contract.symbol, contract.secType, contract.currency, execution.execId, execution.orderId, execution.shares, execution.lastLiquidity)

app = IBapi()
app.connect('127.0.0.1', 7496, 0)

app.nextValidId = 1

while True:
    if hasattr(app, 'nextValidId'):
        break
    else:
        app.nextValidId = 1

# Define contract
contract = Contract()
contract.symbol = 'QQQ'
contract.secType = 'STK'
contract.exchange = 'SMART'
contract.currency = 'USD'

# Define order
buy_order = Order()
buy_order.action = 'BUY'
buy_order.totalQuantity = 1
buy_order.orderType = 'LMT'

sell_order = Order()
sell_order.action = 'SELL'
sell_order.totalQuantity = 1
sell_order.orderType = 'LMT'

# Get the current time in New York timezone (EST)
ny_tz = pytz.timezone('America/New_York')
current_time = datetime.now(ny_tz)

# Define the buy and sell times in New York timezone (EST)
buy_time = current_time.replace(hour=19, minute=45, second=0, microsecond=0)
sell_time = current_time.replace(hour=4, minute=15, second=0, microsecond=0)

# Wait until it's time to place the buy order
while current_time < buy_time:
    current_time = datetime.now(ny_tz)

# Place the buy order
app.placeOrder(app.nextOrderId(), contract, buy_order)

# Wait for the order to fill
while True:
    if app.orderStatus:
        break

# Wait until it's time to place the sell order
while current_time < sell_time:
    current_time = datetime.now(ny_tz)

# Place the sell order
app.placeOrder(app.nextOrderId(), contract, sell_order)

# Wait for the order to fill
while True:
    if app.orderStatus:
        break

app.disconnect()
