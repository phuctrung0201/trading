from source import okx

if __name__ == "__main__":
    client = okx.Client(
        API_KEY="d009e341-3d49-4f55-b198-548281f1f3b5",
        SECRET_KEY="136E730027D0803623471CCCDCD54809",
        PASSPHRASE="Nothing0@0!",
        demo=True,
    )

    # Stream real-time 1-minute candles for ETH-USDT.
    # Blocks until Ctrl-C.
    channel = client.subscribe(instrument="ETH-USDT", bar="1m")
    for candle in channel:
        print(candle)
