# Prompt — Fiche narrative d'un film

Modèle : `gemini-2.5-flash` (thinking désactivé pour la latence, seuil ≤ 5s)

Recherche par **titre** (pas par id interne) : `GET /fiche?titre=...`.
Deux modes selon que le film est trouvé en base Gold ou non.

## Sortie structurée (JSON, `response_schema`)

```json
{
  "accroche": "phrase punchy type bande-annonce",
  "resume": "résumé narratif 100-150 mots, sans spoiler",
  "ambiance": "ton, atmosphère, style visuel/musical",
  "pourquoi_regarder": "argumentaire court, pour quel public"
}
```

Si la 1ère réponse ne respecte pas le schéma JSON, un 2e essai est fait
avec une instruction de correction (cf. `_generate` dans
`api/routers/fiche.py`).

## Mode "catalogue" (film trouvé en base Gold)

Système : critique de cinéma, s'appuie uniquement sur les faits fournis
(titre, année, genres, synopsis TMDB), n'invente aucun détail d'intrigue
absent du synopsis. Si `overview` est NULL (~44% du catalogue enrichi),
reste général sans inventer d'intrigue.

Utilisateur :
```
Titre : {titre}
Année : {annee}
Genres : {genres}
Synopsis (TMDB) : {overview ou "Synopsis non disponible."}

Rédige la fiche à partir de ces informations.
```

## Mode "connaissances LLM" (film absent du catalogue)

Système : répond à partir de ses propres connaissances si le titre est
reconnu ; si incertain, le dit explicitement plutôt que d'inventer une
intrigue plausible. La réponse API inclut `"source": "connaissances_llm"`
pour que le dashboard affiche un disclaimer (donnée non vérifiée).

Utilisateur :
```
Titre recherché : {titre}

Ce film n'est pas dans notre catalogue. Rédige une fiche si tu
reconnais ce titre, sinon indique-le dans le champ 'accroche' et
laisse les autres champs vides.
```
