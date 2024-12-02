#!/bin/bash

# Déterminez le répertoire contenant ce script
#APP_DIR="$(dirname "$0")"
APP_DIR="$(dirname "$0")"

# Chemin de l'environnement virtuel
VENV_PATH="${APP_DIR}/app_ressources/venv_steam_recommender"

# Création de l'environnement virtuel s'il n'existe pas déjà
if [ ! -d "$VENV_PATH" ]; then
    echo "Création de l'environnement virtuel..."
    python3 -m venv "$VENV_PATH"
    echo "Environnement virtuel créé."
    # Installation des dépendances à partir de requirements.txt
    echo "Installation des requirements ..."
    pip install --upgrade pip
    pip install -r "${APP_DIR}/app_ressources/launcher/requirements.txt"
fi

# Activation de l'environnement virtuel
source "${VENV_PATH}/bin/activate"

# Vérification si Streamlit est installé
if ! python -m streamlit --version; then
    echo "Streamlit n'est pas installé. Installation en cours..."
    pip install streamlit
    echo "Streamlit a été installé."
else
    echo "Streamlit est déjà installé."
fi

# Construit le chemin absolu vers le script Python
APP_PATH="${APP_DIR}/app_ressources/Steam_games_recommender.py"

#Donne l'autorisation d'exectuer le fichier
chmod +x "$APP_PATH"
echo "Chemin du script Python : ${APP_PATH}"

#cd ${APP_DIR}/app_ressources
cd "${APP_DIR}/app_ressources"

# Lancez l'application Streamlit
streamlit run "$APP_PATH"

# Désactivation de l'environnement virtuel n'est pas nécessaire dans le script
# car le script se termine ici et l'environnement sera désactivé automatiquement
