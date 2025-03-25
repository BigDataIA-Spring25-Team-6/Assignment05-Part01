import requests
import pandas as pd
import os
import snowflake.connector
from datetime import datetime, timezone
from dotenv import load_dotenv

# Load .env
load_dotenv()

# Fetch QuickFS API key
QUICK_FS_API_KEY = os.getenv("QUICK_FS_API_KEY")

# Fetch data from QuickFS
url = f"https://public-api.quickfs.net/v1/data/all-data/NVDA?api_key={QUICK_FS_API_KEY}"
response = requests.get(url)
data = response.json()

# Create DataFrame
quarterly_data = pd.DataFrame(data['data']['financials']['quarterly'])
quarterly_data['period_end_date'] = pd.to_datetime(quarterly_data['period_end_date'])

# Filter: 2020 to 2025 only
quarterly_data = quarterly_data[
    (quarterly_data['period_end_date'] >= '2020-01-01') &
    (quarterly_data['period_end_date'] <= '2025-12-31')
]

# Compute derived metrics
quarterly_data['trailing_pe'] = (
    quarterly_data['period_end_price'] / quarterly_data['eps_diluted'].rolling(window=4).sum()
)
quarterly_data['ev_to_ebitda'] = (
    quarterly_data['enterprise_value'] / quarterly_data['ebitda']
)

# Final DataFrame structure
df = quarterly_data[[
    'period_end_date',
    'market_cap',
    'enterprise_value',
    'trailing_pe',
    'forward_pe',
    'peg_ratio',
    'price_to_sales',
    'price_to_book',
    'enterprise_value_to_sales',
    'ev_to_ebitda'
]].copy()

df.insert(0, 'symbol', 'NVDA')
df['scraped_at'] = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')
df = df.where(pd.notnull(df), None)

# Connect to Snowflake
conn = snowflake.connector.connect(
    user=os.getenv("SNOWFLAKE_USER"),
    password=os.getenv("SNOWFLAKE_PASSWORD"),
    account=os.getenv("SNOWFLAKE_ACCOUNT"),
    warehouse=os.getenv("SNOWFLAKE_WAREHOUSE"),
    database=os.getenv("SNOWFLAKE_DATABASE"),
    schema=os.getenv("SNOWFLAKE_SCHEMA")
)
cursor = conn.cursor()

# Create table if not exists
cursor.execute("""
CREATE TABLE IF NOT EXISTS quarterly_valuation_metrics (
    symbol STRING,
    period_end_date DATE,
    market_cap FLOAT,
    enterprise_value FLOAT,
    trailing_pe FLOAT,
    forward_pe FLOAT,
    peg_ratio FLOAT,
    price_to_sales FLOAT,
    price_to_book FLOAT,
    enterprise_value_to_sales FLOAT,
    ev_to_ebitda FLOAT,
    scraped_at TIMESTAMP
)
""")

# Upload data
for _, row in df.iterrows():
    cursor.execute("""
        INSERT INTO quarterly_valuation_metrics (
            symbol, period_end_date, market_cap, enterprise_value,
            trailing_pe, forward_pe, peg_ratio, price_to_sales,
            price_to_book, enterprise_value_to_sales, ev_to_ebitda, scraped_at
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """, (
        row['symbol'], row['period_end_date'], row['market_cap'], row['enterprise_value'],
        row['trailing_pe'], row['forward_pe'], row['peg_ratio'],
        row['price_to_sales'], row['price_to_book'],
        row['enterprise_value_to_sales'], row['ev_to_ebitda'], row['scraped_at']
    ))

print("✅ Valuation data uploaded directly to Snowflake.")

cursor.close()
conn.close()