from fastapi import APIRouter, HTTPException, Query
from app.database import (
    DB_PATH,
    get_customer_with_interactions,
    get_leads,
    list_customers,
)
from app.customer_memory import lookup_customer_profile, normalize_phone
from app.models import Customer, CustomerProfile, Interaction, Lead

router = APIRouter()


@router.get("/leads")
async def list_leads():
    rows = await get_leads()
    return [Lead(**r) for r in rows]


@router.get("/reports")
async def list_reports():
    rows = await get_leads()
    return {"reports": [Lead(**r) for r in rows]}


@router.get("/customers")
async def list_customer_profiles(limit: int = Query(default=100, ge=1, le=500)):
    rows = await list_customers(limit=limit)
    return [Customer(**r) for r in rows]


@router.get("/customers/lookup")
async def lookup_customer(
    phone: str | None = Query(default=None),
    email: str | None = Query(default=None),
):
    normalized_phone = normalize_phone(phone)
    profile = await lookup_customer_profile(phone=normalized_phone, email=email)
    if not profile:
        raise HTTPException(status_code=404, detail="Customer not found")
    return CustomerProfile(
        customer=Customer(**profile["customer"]),
        interactions=[Interaction(**i) for i in profile.get("interactions", [])],
    )


@router.get("/customers/{customer_id}")
async def get_customer(customer_id: int):
    profile = await get_customer_with_interactions(customer_id)
    if not profile:
        raise HTTPException(status_code=404, detail="Customer not found")
    return CustomerProfile(
        customer=Customer(**profile["customer"]),
        interactions=[Interaction(**i) for i in profile.get("interactions", [])],
    )


@router.get("/db-path")
async def db_path():
    return {"db_path": DB_PATH}
