# Prompt — Recommandation (formulaire d'envies)

Modèle : `gemini-2.5-flash` (thinking désactivé, seuil ≤ 5s)

`GET /recommandation?duree_max_heures=...&genre=...&note_min=...`

Contrairement à `/fiche` et `/comparaison`, cet endpoint est **100% basé
sur les connaissances de Gemini** — pas de requête sur le catalogue Gold.
L'utilisateur exprime ses envies via un formulaire structuré plutôt qu'un
historique de notation :

- `duree_max_heures` (float) : durée maximale souhaitée
- `genre` (str) : genre recherché
- `note_min` (float, 0-10) : note minimale souhaitée

## Sortie structurée (JSON, `response_schema`)

```json
{
  "suggestions": [
    {
      "titre": "...",
      "annee": 2017,
      "duree_heures": 1.95,
      "note_estimee": 7.4,
      "justification": "..."
    }
  ]
}
```

Au moins 5 suggestions requises. Si la 1ère réponse en contient moins,
un 2e essai est fait avec un rappel explicite du nombre manquant.

## Système

Conseiller cinéma qui recommande des films réels (jamais inventés),
en français, à partir de ses connaissances, en respectant les critères
(durée maximale, genre, note minimale), avec justification concrète liée
à chaque critère.

## Utilisateur

```
Durée souhaitée : {duree_max_heures}h maximum
Genre recherché : {genre}
Note minimale souhaitée : {note_min}/10

Propose au moins 5 films réels qui correspondent le mieux possible à
ces critères.
```
