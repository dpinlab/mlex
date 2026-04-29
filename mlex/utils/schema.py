from pydantic import (
    BaseModel,
    StrictInt,
    ConfigDict,
    ValidationInfo,
    field_validator,
    model_validator,
)
from typing import Literal
from datetime import datetime
from decimal import Decimal


class TransactionFlow(BaseModel):
    tx_index: StrictInt
    balance_amount: StrictInt
    tx_value: StrictInt
    timestamp: datetime

    @model_validator(mode="before")
    @classmethod
    def transform_transaction_flow(cls, data: dict):
        decimal_balance_amount = Decimal(str(data.get("balance_amount", 0)))
        decimal_tx_value = Decimal(str(data.get("tx_value", 0)))
        integer_balance_amount = int((decimal_balance_amount * 100).to_integral_value())
        integer_tx_value = int((decimal_tx_value * 100).to_integral_value())

        data.update(
            {
                "balance_amount": integer_balance_amount,
                "tx_value": integer_tx_value,
            }
        )
        return data


class Transaction(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    index: int
    account_id: str
    balance_type: str
    timestep: int
    balance_amount: int
    previous_balance: int

    @field_validator("balance_amount", "previous_balance", mode="before")
    @classmethod
    def balance_conversion(cls, v, info: ValidationInfo):
        return int(float(v) * 100)
