import os
import pandas as pd
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()

DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST","localhost")
DB_PORT = os.getenv("DB_PORT","5432")
DB_NAME = os.getenv("DB_NAME", "retail_db")

raw_data_path = os.path.join("data","raw","online_retail_II.xlsx")

def get_db_engine():
    connection_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    return create_engine(connection_url)

def load_and_standardize_data(file_path:str) -> pd.DataFrame:
    print(f" [1/3] reading excel sheets from {file_path}")

    excel_file = pd.ExcelFile(file_path)
    sheet_dfs = []

    for sheet_name in excel_file.sheet_names:
        print(f"loading the sheet {sheet_name}")
        df_sheet = pd.read_excel(excel_file, sheet_name=sheet_name)
        sheet_dfs.append(df_sheet)
    df_raw = pd.concat(sheet_dfs, ignore_index=True)
    print(f"total rows loaded: {len(df_raw):,}")

    #standardizing the column names to snakecase
    df_raw.columns = [col.strip().lower().replace(" ","_").replace("customer_id","customer_id") for col in df_raw.columns]

    rename_map = { "invoice":"invoice_no", "stockcode":"stock_code", "invoicedate":"invoice_date",
                 "price":"unit_price", "customer_id":"customer_id","country":"country","description":"description",
                 "quantity":"quantity"}

    df_raw = df_raw.rename(columns = {k:v for k, v in rename_map.items() if k in df_raw.columns})

    return df_raw


def ingest_to_postgres():
    if not os.path.exists(raw_data_path):
        raise FileNotFoundError(f" dataset not found ar {raw_data_path}")
    df = load_and_standardize_data(raw_data_path)

    print(" connecting to postgresql")
    engine = get_db_engine()

    print('inserting raw data into the raw_transactions table')

    #using chunksize to prevent high memory usage during insertion
    df.to_sql( name='raw_transactions', con=engine, if_exists="replace", index=False,
              chunksize=50000, method="multi")
    print('ingestion complete table raw_transaactions is ready in postgresql')

if __name__ == "__main__":
    ingest_to_postgres()

