from fastapi import APIRouter
from app.database import get_leads
from app.models import Lead

router = APIRouter()


@router.get("/leads")
async def list_leads():
    rows = await get_leads()
    return [Lead(**r) for r in rows]
