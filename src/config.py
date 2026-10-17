#central place to store the db connection logic (to reduce repeatition in every script)

import os 
from dotenv import load_dotenv
from sqlalchemy import create_engine

load_dotenv() #to load env variables 

#constructing db url 
DB_USER = os.getenv("DB_USER")
DB_PASSWORD = os.getenv("DB_PASSWORD")
DB_HOST = os.getenv("DB_HOST","localhost")
DB_PORT = os.getenv("DB_PORT","5432")
DB_NAME = os.getenv("DB_NAME", "retail_db")

DB_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

#reuable engine 
engine = create_engine(DB_URL)