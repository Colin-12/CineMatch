# Prompt — Comparaison de deux films

Modèle : `gemini-2.5-flash` (thinking désactivé, seuil ≤ 8s)

`GET /comparaison?titre_1=...&titre_2=...` — chaque film est résolu
indépendamment (`resolve_film`, voir `fiche.md`) : "catalogue" si trouvé
en Gold, "connaissances_llm" sinon. Les deux modes peuvent être mélangés
dans un seul appel (ex. un film catalogué + un film récent absent).

## Sortie structurée (JSON, `response_schema`)

```json
{
  "points_communs": "...",
  "differences_cles": "...",
  "note_comparative": "..."
}
```

## Système

Compare deux films pour aider un spectateur à choisir. Pour un film
"catalogue", s'appuie uniquement sur les faits fournis (titre, année,
genres, synopsis, note TMDB réelle). Pour un film "connaissances_llm",
répond depuis ses propres connaissances si le titre est reconnu, le dit
explicitement sinon.

## Utilisateur

```
Film A : {titre} ({annee})
Genres : {genres}
Note TMDB : {note_tmdb ou "non disponible"}
Synopsis : {overview ou "Synopsis non disponible."}

Film B : {idem, ou "{titre} (absent de notre catalogue)" si non trouvé}

Compare ces deux films : points communs, différences clés, et une note
comparative (qui l'emporte sur quels aspects, en s'appuyant sur les
notes TMDB réelles quand disponibles).
```
