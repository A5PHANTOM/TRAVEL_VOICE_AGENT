from fastapi import APIRouter
from app.database import DB_PATH, get_leads
from app.models import Lead

router = APIRouter()


@router.get("/leads")
async def list_leads():
    rows = await get_leads()
    return [Lead(**r) for r in rows]


@router.get("/reports")
async def list_reports():
    rows = await get_leads()
    return {"reports": [Lead(**r) for r in rows]}


@router.get("/db-path")
async def db_path():
    return {"db_path": DB_PATH}
