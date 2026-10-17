import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors
from src.config import engine
from src.validation.schemas import raw_transaction_schema

def extract_and_vaidate():
    print("[1/3] extrating the data from postgreSQL")
    df = pd.read_sql("SELECT * FROM raw_transactions", con=engine)
    #using the reuable engine from config.py
    print(f"loaded {len(df):,} rows")

    print("[2/3] preparing the data types")
    df['invoice_date'] = pd.to_datetime(df['invoice_date'])
    df['invoice_no'] = df['invoice_no'].astype(str)
    df['stock_code'] = df['country'].astype(str)

    print("[3/3] running pandera validation")
    try:
        #lazy = true, rather than failing at first error it ensures panderas collects all errors before failing and display at end
        validated_df = raw_transaction_schema.validate(df, lazy=True)
        print('validation passed, raw data matched our expected schema')
        return validated_df
    except SchemaErrors as err:
        print('validation failed, found data quality issues')
        print(err.failure_cases.head(25))
        return err.data   # returns the df anyway to inspect 

if __name__ == "__main__":
    extract_and_vaidate()

    