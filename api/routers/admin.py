"""Endpoint /admin/stats : chiffres clés Gold pour la vue admin du dashboard."""

from fastapi import APIRouter

from api.db import fetch_admin_stats

router = APIRouter(prefix="/admin", tags=["admin"])


@router.get("/stats")
def get_admin_stats() -> dict:
    """Compteurs, taux de couverture et fraîcheur des données Gold."""
    return fetch_admin_stats()
