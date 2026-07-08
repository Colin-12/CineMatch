from pydantic import BaseModel


class Film(BaseModel):
    film_id: int
    titre: str
    annee: int
    genres: list[str]


class Utilisateur(BaseModel):
    user_id: int
    age: int | None = None


class Notation(BaseModel):
    user_id: int
    film_id: int
    note: float
    timestamp: int


class Avis(BaseModel):
    user_id: int
    film_id: int
    texte: str
    timestamp: int


class SentimentScore(BaseModel):
    film_id: int
    score: float
    label: str


class PredictionNote(BaseModel):
    user_id: int
    film_id: int
    note_predite: float
