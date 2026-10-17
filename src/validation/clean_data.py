import pandas as pd
from src.config import engine
from src.validation.schemas import clean_transaction_schema

NON_PRODUCT_CODES = {"POST", "D","M","BANK CHARGES", "PADS", "DOT", "CRUK", "B", "ADJUST", "ADJUST2"}

def clean_and_store_transactions():
    print("[1/5] reading raw transactions from PostgreSQL")
    df = pd.read_sql("SELECT * FROM raw_transactions", con=engine)
    intial_count = len(df)
    print(f" intial rows: {intial_count}")

    print("[2/5] applying data cleaning rules")

    #1. drop rows without customer_id since we cannnot track anonymous guests across lifecycle
    df = df.dropna(subset=['customer_id']).copy()

    #2. remove exact dupliacte records
    df = df.drop_duplicates().copy()


    #3. filter out bad debt adjustments and zero/negative prices
    df = df[df['unit_price'] > 0.0].copy()
    df = df[df['quantity'] != 0].copy()


    #4. standadize the datatypes
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    df['invoice_no'] = df['invoice_no'].astype(str).str.strip()
    df['stock_code'] = df['stock_code'].astype(str).str.strip().str.upper()
    df['description'] = df['description'].fillna('UNKNOWN').astype(str).str.strip()
    df['customer_id'] = df['customer_id'].astype(int)
    df['country'] = df['country'].astype(str).str.strip()

    #5. remove non product stock codes (postage, manual adjustments, bank charges)
    df = df[~df['stock_code'].isin(NON_PRODUCT_CODES)].copy()


    print('[3/5] engineering transaction level flags')
    #flag cancelled orders ( ionvoice starts with c or quantity is negative)
    df['is_cancelled'] = df['invoice_no'].str.upper().str.startswith('C') | (df['quantity']<0)
    df['line_total'] = (df['quantity'] * df['unit_price']).round(2)


    print(f" cleaned rows: {len(df):,} (dropped {intial_count - len(df):,} invalid/anonymous rows)")

    print("[4/5] validating cleaned dataset with pandera")
    validated_df = clean_transaction_schema.validate(df, lazy=True)
    print('validation passsed!')

    print('[5/5] saving clean transactions table to postgresql')
    validated_df.to_sql(
        name='clean_transactions', con=engine, if_exists='replace', index=False, chunksize=50000, method='multi'
    )
    print('done, table clean_transactions is ready for phase 3 feature engineering')

if __name__ == "__main__":
    clean_and_store_transactions()