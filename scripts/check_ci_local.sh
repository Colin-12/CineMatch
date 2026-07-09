#!/usr/bin/env bash
#
# scripts/check_ci_local.sh
#
# Reproduit localement, a l'identique, le job CI defini dans
# .github/workflows/ci.yml : clone la branche courante depuis le depot LOCAL
# (pas depuis le remote GitHub) dans un dossier temporaire, cree un venv
# frais, installe requirements.txt, puis lance flake8 et pytest dans le meme
# ordre que la CI (job "lint" puis job "test").
#
# Cloner depuis le depot local (et non depuis origin) garantit qu'on teste
# exactement l'etat committe de la branche courante : les modifications non
# committees ne sont PAS incluses, tout comme la vraie CI qui ne voit que ce
# qui a ete pousse/committe.
#
# S'arrete au premier echec avec un message clair indiquant l'etape fautive.
#
# Usage : ./scripts/check_ci_local.sh
# Variable optionnelle : CI_CHECK_DIR (defaut : /tmp/ci-check)

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CI_CHECK_DIR="${CI_CHECK_DIR:-/tmp/ci-check}"
CLONE_DIR="$CI_CHECK_DIR/repo"
VENV_DIR="$CI_CHECK_DIR/venv"

STEP="initialisation"
RESULT="FAIL"

cleanup() {
    # Dossier conserve en cas d'echec (pour inspection), supprime si tout est vert.
    if [ "$RESULT" = "PASS" ]; then
        rm -rf "$CI_CHECK_DIR"
    fi
}
trap cleanup EXIT

fail() {
    echo ""
    echo "=========================================="
    echo " RESUME CI LOCALE : FAIL"
    echo " Etape en echec : $STEP"
    echo "=========================================="
    echo "$1"
    echo ""
    echo "Dossier conserve pour inspection : $CI_CHECK_DIR"
    exit 1
}

echo "=========================================="
echo " Verification CI locale (reproduit ci.yml)"
echo "=========================================="

STEP="detection de la branche courante"
BRANCH="$(git -C "$REPO_ROOT" rev-parse --abbrev-ref HEAD)" \
    || fail "Impossible de determiner la branche courante (repo Git valide ?)."
echo "Branche courante   : $BRANCH"
echo "Depot local source : $REPO_ROOT"

STEP="preparation du dossier temporaire"
rm -rf "$CI_CHECK_DIR"
mkdir -p "$CI_CHECK_DIR"

STEP="clonage local (branche $BRANCH, depuis le depot local, pas le remote)"
git clone --local --no-hardlinks --branch "$BRANCH" --single-branch \
    "$REPO_ROOT" "$CLONE_DIR" \
    || fail "Le clone local de la branche '$BRANCH' a echoue."
echo "Clone               : $CLONE_DIR"

STEP="recherche d'un interpreteur Python"
# `command -v` seul ne suffit pas : sous Windows, `python3` peut resoudre vers
# le stub Microsoft Store (present dans le PATH mais non fonctionnel), d'ou la
# verification effective via `--version` pour chaque candidat.
PYTHON_CMD=()
for candidate in python3 python; do
    if command -v "$candidate" >/dev/null 2>&1 && "$candidate" --version >/dev/null 2>&1; then
        PYTHON_CMD=("$candidate")
        break
    fi
done
if [ ${#PYTHON_CMD[@]} -eq 0 ] && command -v py >/dev/null 2>&1 && py -3 --version >/dev/null 2>&1; then
    PYTHON_CMD=(py -3)
fi
if [ ${#PYTHON_CMD[@]} -eq 0 ]; then
    fail "Aucun interpreteur Python fonctionnel trouve (python3 / python / py -3)."
fi
echo "Interpreteur Python : ${PYTHON_CMD[*]} ($("${PYTHON_CMD[@]}" --version 2>&1))"

STEP="creation du venv frais"
"${PYTHON_CMD[@]}" -m venv "$VENV_DIR" || fail "La creation du venv a echoue."

if [ -f "$VENV_DIR/Scripts/python.exe" ]; then
    VENV_PYTHON="$VENV_DIR/Scripts/python.exe"
else
    VENV_PYTHON="$VENV_DIR/bin/python"
fi

STEP="installation de requirements.txt"
"$VENV_PYTHON" -m pip install --quiet --upgrade pip \
    || fail "La mise a jour de pip dans le venv a echoue."
"$VENV_PYTHON" -m pip install --quiet -r "$CLONE_DIR/requirements.txt" \
    || fail "L'installation de requirements.txt a echoue."

STEP="flake8 (job 'lint' de la CI)"
if ! (cd "$CLONE_DIR" && "$VENV_PYTHON" -m flake8 .); then
    fail "flake8 a signale des erreurs de lint sur l'etat committe de '$BRANCH'."
fi
echo "flake8 : OK"

STEP="pytest (job 'test' de la CI)"
if ! (cd "$CLONE_DIR" && "$VENV_PYTHON" -m pytest); then
    fail "pytest a echoue (au moins un test rouge) sur l'etat committe de '$BRANCH'."
fi
echo "pytest : OK"

RESULT="PASS"
echo ""
echo "=========================================="
echo " RESUME CI LOCALE : PASS"
echo " flake8 + pytest OK sur la branche '$BRANCH' (etat committe)"
echo "=========================================="
