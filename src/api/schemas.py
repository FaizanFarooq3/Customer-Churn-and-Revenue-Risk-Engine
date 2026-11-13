from pydantic import BaseModel, Field

class CustomerFeaturesRequest(BaseModel):
    #it is the incoming json payload expected by the api
    frequency: float = Field(..., description="total no of historical orders")
    monetary: float = Field(..., description="Total lifetime spend")
    total_quantity: float = Field(..., description="total items purchased")
    unique_products: float = Field(..., description="count of distinct products bougth")
    recency_days: float = Field(..., description="Days since the last purchase")
    tenure_days: float = Field(..., description="Days since first purchase")
    avg_order_value: float = Field(..., description="avg value per order")
    avg_basket_size: float = Field(..., description="avg items per order")
    purchase_frequency_per_month: float = Field(..., description="orers per month")
    avg_days_between_orders: float = Field(..., description="days between consecutive orders")
    is_uk: int = Field(..., description="1 if customer is in uk or 0 otherwise")
    cancellation_rate: float = Field(..., description="ratio of cancelled to total invoices")
    spend_last_90d: float = Field(..., description="total spend in the 90 days before cutoff")
    spend_prev_90d: float = Field(..., description="total spend between 91 and 180 days")
    spending_trend_90d: float = Field(..., description="ratio of recent spend to previous spend")

class RiskPredictionResponse(BaseModel):
    """the json payload returned by the api"""
    risk_probability: float
    risk_band: str
    expected_revenue_at_risk: float
    customer_segment: str
    recommended_action: str
    model_version: str