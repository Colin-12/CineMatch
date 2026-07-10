from fastapi import FastAPI

from api.routers import admin, comparaison, fiche, prediction, recommandation, sentiment

app = FastAPI(title="CineMatch API")

app.include_router(fiche.router)
app.include_router(comparaison.router)
app.include_router(recommandation.router)
app.include_router(prediction.router)
app.include_router(sentiment.router)
app.include_router(admin.router)


@app.get("/health")
def health() -> dict[str, str]:
    """Vérifie que l'API répond."""
    return {"status": "ok"}
