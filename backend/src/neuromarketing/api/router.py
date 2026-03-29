"""Aggregated API router."""

from fastapi import APIRouter

from neuromarketing.api.analysis import router as analysis_router
from neuromarketing.api.health import router as health_router

api_router = APIRouter()
api_router.include_router(health_router)
api_router.include_router(analysis_router)
