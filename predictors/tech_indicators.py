import pandas as pd

import numpy as np
# 1. SMA 

def calculate_sma(df, window):
    '''
    This function calculates the simple moving average (SMA) of closing prices
    for multiple stock symbols over a given time window.
    
    Parameters:
    df (Pandas DataFrame): The input DataFrame containing the stock data
    symbols (list): A list of stock symbols for which to calculate the SMA
    window (int): The time window over which to calculate the SMA
    
    Returns:
    sma_df (Pandas DataFrame): A DataFrame containing the SMA of closing prices
    for all the specified stock symbols and time window.
    '''
    
    sma_df = df  # Initialize an empty DataFrame to store the results
    
    
    sma_df["SMA_"+str(window)] = df["close"].rolling(window=window).mean()
    
    return sma_df[ "SMA_"+str(window)]



# 2. EMA 
def calculate_ema(df,  window):
    '''
    This function calculates the simple moving average (SMA) of closing prices
    for multiple stock symbols over a given time window.
    
    Parameters:
    df (Pandas DataFrame): The input DataFrame containing the stock data
    symbols (list): A list of stock symbols for which to calculate the SMA
    window (int): The time window over which to calculate the SMA
    
    Returns:
    sma_df (Pandas DataFrame): A DataFrame containing the SMA of closing prices
    for all the specified stock symbols and time window.
    '''
    
    sma_df = df  # Initialize an empty DataFrame to store the results
    

    sma_df["EMA"] = df["close"].ewm(span=window, adjust=False).mean()
    
    return sma_df["EMA"]



# 3. STOCH
def calculate_stoch(symbol_df, k_window=14, d_window=3): 
    
    # Filter the DataFrame to include only the rows with the specified symbol
    #symbol_df = get_daily_adjusted_prices(symbol)
    
    # Set the date column as the DataFrame index
    symbol_df = symbol_df
    symbol_df['close'] = pd.to_numeric(symbol_df['close'])

    # Calculate the lowest and highest closing prices over the past k_window days
    low_min = symbol_df['low'].rolling(window=k_window).min()
    high_max = symbol_df['high'].rolling(window=k_window).max()
    
    # Calculate the %K line using the current closing price and the recent trading range
    k = 100 * (symbol_df['close']) - low_min / (high_max - low_min)
    
    # Calculate the %D line using a moving average of the %K line
    d = k.rolling(window=d_window).mean()
    
    # Combine the %K and %D lines into a DataFrame
    stoch_df = pd.DataFrame({'%K': k, '%D': d})
    
    return k, d



# 4. RSI

def calculate_rsi(df, period=14):
    """
    Calculates the relative strength index (RSI) of a DataFrame.

    Parameters:
    period (int): Number of periods to use for the RSI calculation.

    Returns:
    pandas.DataFrame: DataFrame containing the RSI values.
    """
    #df = get_daily_adjusted_prices(symbol)
    df["close"] = pd.to_numeric(df["close"])
    delta = df['close'].diff()

    # Get upward and downward price movements
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    # Calculate the exponential moving average (EMA) of gains and losses
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()

    # Calculate the relative strength (RS)
    rs = avg_gain / avg_loss

    # Calculate the RSI
    rsi = 100 - (100 / (1 + rs))

    # Add RSI values to the DataFrame
    df['rsi'] = rsi

    return df.loc[:,'rsi']


# 5. MFI

def calculate_mfi(df,period=14):
    """
    Calculates the money flow index (MFI) of a DataFrame.

    Parameters:
    period (int): Number of periods to use for the MFI calculation.

    Returns:
    pandas.DataFrame: DataFrame containing the MFI values.
    """
    #df = get_daily_adjusted_prices(symbol)
    df["close"] = pd.to_numeric(df["close"])
    df["low"] = pd.to_numeric(df["low"])
    df["high"] = pd.to_numeric(df["high"])
    df["volume"] = pd.to_numeric(df["volume"])
    # Calculate typical price
    typical_price = (df['close'] + df['low'] + df['high']) / 3

    # Calculate raw money flow
    money_flow = typical_price * df['volume']

    # Get positive and negative money flow
    positive_flow = np.where(typical_price > typical_price.shift(1).fillna(method='bfill'), money_flow, 0)
    negative_flow = np.where(typical_price < typical_price.shift(1).fillna(method='bfill'), money_flow, 0)
   
    # Calculate the money flow ratio
    raw_mfr = positive_flow / negative_flow
    mfr = pd.Series(np.where(negative_flow == 0, 100, 100 - (100 / (1 + raw_mfr))))

    # Calculate the MFI
    mfi = mfr.rolling(window=period).mean()

    # Add MFI values to the DataFrame
    df['mfr'] = mfr
    df['mfi'] = mfi
    df['negative_flow'] = negative_flow
    df['positive_flow'] = positive_flow

    return df.loc[:,['mfr','negative_flow','positive_flow']]



# 6. SAR 

def calculate_sar(df, lookback=14, acceleration_factor=0.02):
    # Load the price data into a pandas DataFrame
    #df = get_daily_adjusted_prices(symbol)
   
    #df["close"] = pd.to_numeric(df["close"])
    #df["low"] = pd.to_numeric(df["low"])
    #df["high"] = pd.to_numeric(df["high"])
    
    # Initialize the SAR for the first day as the lowest low over the lookback period
    sar = df['low'].rolling(window=lookback).min()
    trend = pd.Series(np.zeros(len(df)))
   
    # Loop through the data and calculate the SAR and trend for each day
    for i in range(lookback, len(df)):
        if df['low'][i] > sar[i-1]:
            trend[i] = 1
        elif df['high'][i] < sar[i-1]:
            trend[i] = -1
        else:
            trend[i] = trend[i-1]

        if trend[i] == 1:
            if sar[i-1] == df['low'][i-1]:
                sar[i] = sar[i-1] + acceleration_factor * (df['high'][i-1] - sar[i-1])
            else:
                sar[i] = sar[i-1] + acceleration_factor * (df['high'][i-1] - sar[i-1]) * trend[i-1]
            if sar[i] > df['low'][i]:
                sar[i] = df['low'][i]
        else:
            if sar[i-1] == df['high'][i-1]:
                sar[i] = sar[i-1] - acceleration_factor * (sar[i-1] - df['low'][i-1])
            else:
                sar[i] = sar[i-1] + acceleration_factor * (df['low'][i-1] - sar[i-1]) * trend[i-1]
            if sar[i] < df['high'][i]:
                sar[i] = df['high'][i]
        
    
    # Create a new DataFrame with the calculated Parabolic SAR values and the original price data
    sar_df = pd.DataFrame({'Parabolic SAR': sar}, index=df['Date'])
    return sar
    
   
"""
AAPL = pd.read_csv("stock_price/Daily/AAPL.csv")
df = AAPL.copy()
sar = calculate_sar(df)
print(sar)

"""





def calc_AD(df):
    # Initialize the AD line as the closing price
    df["close"] = pd.to_numeric(df["close"])
    df["low"] = pd.to_numeric(df["low"])
    df["high"] = pd.to_numeric(df["high"])
    df["volume"] = pd.to_numeric(df["volume"])
    ad = df['close'].copy()
    
    # Calculate the money flow multiplier (MFM)
    mfm = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    
    # Calculate the money flow volume (MFV)
    mf_volume = mfm * df['volume']
    
    # Calculate the cumulative sum of the money flow volume
    cmf = mf_volume.cumsum()
    
    # Calculate the AD line
    ad.iloc[0] = cmf.iloc[0]
    for i in range(1, len(ad)):
        if df['close'].iloc[i] > df['close'].iloc[i-1]:
            ad.iloc[i] = ad.iloc[i-1] + cmf.iloc[i]
        elif df['close'].iloc[i] < df['close'].iloc[i-1]:
            ad.iloc[i] = ad.iloc[i-1] - cmf.iloc[i]
        else:
            ad.iloc[i] = ad.iloc[i-1]
            
    # Set the index of the DataFrame to the date column
    #ad.index = df['Date']
    
    return ad



