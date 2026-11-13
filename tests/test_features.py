import pandas as pd
from src.features.build_features import build_customer_snapshot

def test_build_customer_snapshot_no_temporal_leakage():
    """used to verify features only use transactions before cutoff_date
    and target (is_at_risk) only uses transactions after the cutoff-date"""

    synthetic_df = pd.DataFrame(
        {
            "invoice_no":["INV1", "INV2", "INV3","INV4"],
            "stock_code": ["P1","P2","P1","P3"],
            "quantity": [2,1,5,1],
            "invoice_date": pd.to_datetime([
                    "2011-08-01",  # Customer 101: Before cutoff (historical)
                    "2011-08-15",  # Customer 101: Before cutoff (historical)
                    "2011-09-10",  # Customer 101: After cutoff (future window -> active!)
                    "2011-07-01",  # Customer 202: Before cutoff (historical only -> churns!) 
                    ]),
            "unit_price": [10.0,20.0,10.0,50.0],
            "customer_id": [101,101,101,202],
            "country": ["United Kingdom", "United Kingdom", "United Kingdom", "France"],
            "is_cancelled":[False, False, False, False],
            "line_total": [20.0,20.0,50.0,50.0],
        }
    )
    snapshot = build_customer_snapshot(
        df=synthetic_df, cutoff_date_str="2011-09-01",
        split_name="test",horizon_days=90,)
    
    c101 = snapshot[snapshot['customer_id'] == 101].iloc[0]
    c202 = snapshot[snapshot['customer_id'] == 202].iloc[0]

    #customer 101 had £40 historical spend across 2 invoices BEFORE 2011-09-01
    # (The £50 spend on 2011-09-10 must NOT leak into historical 'monetary')

    assert c101['frequency'] ==2
    assert c101['monetary'] == 40.0
    assert c101['future_90d_revenue'] == 50.0
    assert c101['is_at_risk'] == 0

    #customer 202 has £50 historical spend and £0 future spend -> at risk (1)
    assert c202['frequency'] ==1
    assert c202["monetary"] == 50.0
    assert c202["future_90d_revenue"] == 0.0
    assert c202["is_at_risk"] == 1
    assert c202["is_uk"] == 0

