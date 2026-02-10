from source import okx

if __name__ == "__main__":
    client = okx.Client(
        api_key="d009e341-3d49-4f55-b198-548281f1f3b5",
        secret_key="136E730027D0803623471CCCDCD54809",
        passphrase="Nothing0@0!",
        demo=True,
    )

    # Stream real-time 1-minute candles for ETH-USDT.
    # Blocks until Ctrl-C.
    channel = client.subscribe(instrument="ETH-USDT", bar="1m")
    for candle in channel:
        print(candle)
