#used to define expected shape and rules of our raw data ( schema )
import pandera.pandas as pa
from pandera.pandas import Column, Check

#schema for raw data excatly as it comes from the db
raw_transaction_schema = pa.DataFrameSchema(
    {
        "invoice_no": Column(str, nullable=False),
        "stock_code": Column(str, nullable=False),
        "description": Column(str, nullable=True),
        "quantity": Column(int,nullable=False),
        #quantity can contain -ve values so we are not restricting it to >0

        "invoice_date": Column("datetime64[ns]", nullable=False),
        "unit_price": Column(float, nullable=False, checks=[
            Check.greater_than_or_equal_to(0.0, error='price cannot be negative')
        ]),
        "customer_id": Column(float, nullable=True), #float since pd casts ints with nans to float
        "country": Column(str, nullable=False),

    },
    strict=False,  #allows extra columns if exist but validates the above mentioned
)

#schema for clean/processed data
clean_transaction_schema = pa.DataFrameSchema(
    {
        "invoice_no": Column(str, nullable=False),
        "stock_code": Column(str, nullable=False),
        "description": Column(str, nullable=True),
        "quantity": Column(int, nullable=False, checks=[Check.not_equal_to(0, error='quantity cannot be equal to 0')]),
        "invoice_date": Column("datetime64[ns]", nullable=False),
        "unit_price": Column(float, nullable=False, checks=[Check.greater_than(0.0, error='unit price must be strictly positive')]),
        "customer_id": Column(int, nullable=False),
        "country": Column(str, nullable=False),
        "is_cancelled": Column(bool, nullable=False),
        "line_total": Column(float, nullable=False),
    },strict=False,
)
