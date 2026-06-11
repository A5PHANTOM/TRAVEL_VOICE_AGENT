from pydantic import BaseModel
from typing import Any


class Lead(BaseModel):
    id: int | None = None
    package: str
    lead: dict[str, Any] | None = None
    timestamp: str | None = None


class Customer(BaseModel):
    id: int
    phone: str | None = None
    email: str | None = None
    name: str | None = None
    preferred_accommodation: str | None = None
    preferred_duration_days: int | None = None
    flight_preference: bool | None = None
    budget_range: str | None = None
    destinations: list[str] = []
    preferences: dict[str, Any] = {}
    created_at: str
    updated_at: str


class Interaction(BaseModel):
    id: int
    customer_id: int
    lead_id: int | None = None
    call_id: str | None = None
    destination: str | None = None
    duration_days: int | None = None
    accommodation: str | None = None
    flight_needed: bool | None = None
    summary: str | None = None
    timestamp: str


class CustomerProfile(BaseModel):
    customer: Customer
    interactions: list[Interaction] = []
