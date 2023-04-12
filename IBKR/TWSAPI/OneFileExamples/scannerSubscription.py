from ibapi.client import *
from ibapi.wrapper import *
from ibapi.tag_value import *

port = 7497

class TestApp(EClient, EWrapper):
    def __init__(self):
        EClient.__init__(self, self)

    def nextValidId(self, orderId: int):
        sub = ScannerSubscription()
        sub.instrument = "STK"
        sub.locationCode = "STK.US.MAJOR"
        sub.scanCode = "TOP_OPEN_PERC_GAIN"

        scan_options = []
        filter_options = [
            TagValue("volumeAbove","10000"),
            TagValue("marketCapBelow1e6", "1000"),
            TagValue("priceAbove", '1')
        ]

        self.reqScannerSubscription(orderId, sub, scan_options, filter_options)

    def scannerData(self, reqId, rank, contractDetails, distance, benchmark, projection, legsStr):
        print(f"scannerData. reqId: {reqId}, rank: {rank}, contractDetails: {contractDetails}, distance: {distance}, benchmark: {benchmark}, projection: {projection}, legsStr: {legsStr}.")

    def scannerDataEnd(self, reqId):
        print("ScannerDataEnd!")
        self.cancelScannerSubscription(reqId)
        self.disconnect()


app = TestApp()
app.connect("127.0.0.1", port, 1001)
app.run()