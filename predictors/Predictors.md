# STA561D Probabilistic Machine Learning Project
### 1. Daily Adjusted
- Daily Open/High/Low/Close/Volume
- Weekly Open/High/Low/Close/Volume
- Monthly Open/High/Low/Close/Volume


### 2. Fundamental related data 
#### Income Statement (INCOME_STATEMENT / Quarterly)
- grossProfit
- operatingIncome
- totalRevenue
- researchAndDevelopment

#### Balanced Sheet (BALANCE_SHEET / Quarterly)
- inventory
- totalCurrentAssets
- shortTermDebt
- longTermDebt
- totalLiabilities
- currentNetReceivables
- currentAccountsPayable
- commonStockSharesOutstanding
- totalShareholderEquity


#### Cash Flow（CASH_FLOW / Quarterly）
- operatingCashflow
- cashflowFromFinancing
- cashflowFromInvestment


### 3. Macroeconomic Data
- REAL_GDP: annual data type 
- REAL_GDP_PER_CAPITA: quarterly data type 
- TREASURY_YIELD: short time(3-month: risk free rate)/ long time (30 years) 
- FEDERAL_FUNDS_RATE: Board of Governors of the Federal Reserve System (US), Federal Funds Effective Rate, retrieved from FRED, Federal Reserve Bank of St. Louis
- CPI: monthly data 
- INFLATION: annual data 
- UNEMPLOYMENT: The unemployment rate represents the number of unemployed as a percentage of the labor force.  
- NONFARM_PAYROLL: the monthly US All Employees, a measure of the number of U.S. workers in the economy that excludes proprietors, private household employees, unpaid volunteers, farm employees, and the unincorporated self-employed.


### 4. Technical Indicators
- SMA: simple moving average of closing price
- EMA: exponential moving average of closing price
- VWAP: the volume weighted average for intra
- MACD: the movign average convergence/divergence values
- STOCH: the stochastic oscillator values where here the close is in relation to the recent trading range
- RSI: the relative strength index (RSI) values, where calculates a ratio of the recent upward price movements to the absolute price movement
- MFI: the money flow index values - look for divergence with price to signal reversals. 
- SAR: returns the parabolic SAR values
- AD: The Accumulation/Distribution Line is similar to the On Balance Volume (OBV), which sums the volume times +1/-1 based on whether the close is higher than the previous close.The Accumulation/Distribution Line is interpreted by looking for a divergence in the direction of the indicator relative to price. If the Accumulation/Distribution Line is trending upward it indicates that the price may follow. Also, if the Accumulation/Distribution Line becomes flat while the price is still rising (or falling) then it signals an impending flattening of the price.