from pydantic import BaseModel, ConfigDict, ValidationInfo, field_validator


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
