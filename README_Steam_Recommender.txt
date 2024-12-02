
# Steam Recommender README

Bienvenue dans le README de Steam Recommender, une application conçue pour vous recommander des jeux Steam en fonction de vos préférences.

## Prérequis

- macOS 10.15 (Catalina) ou ultérieur.
- Python 3.8 ou supérieur doit être installé sur votre machine.
- Accès à un terminal ou une invite de commande.

## Installation

### Option 1 : Exécution via le script Bash

1. **Téléchargez l'application** : Téléchargez le dossier de l'application depuis le lieu de partage fourni et extrayez-le dans votre répertoire de choix.

2. **Exécutez le script** : Double-cliquez sur le fichier `Recommender_Steam.sh` situé dans le répertoire principal de l'application. Ce script va automatiquement :
    - Créer un environnement virtuel Python s'il n'existe pas déjà.
    - Installer toutes les dépendances nécessaires.
    - Configurer les permissions nécessaires.
    - Lancer l'application Streamlit dans votre navigateur par défaut.

### Option 2 : Exécution manuelle

1. **Ouvrez un terminal intégré** : Naviguez vers le dossier `Final_App/app_ressources`.

2. **Activez l'environnement virtuel** (si déjà créé) :

    ```
    source venv_steam_recommender/bin/activate
    ```

3. **Lancez l'application** : Exécutez la commande suivante :

    ```
    streamlit run Steam_games_recommender.py
    ```

## Utilisation

Une fois que l'application est lancée, soit via le script soit manuellement, votre navigateur par défaut s'ouvrira automatiquement avec l'application Streamlit chargée. Suivez les instructions à l'écran pour commencer à recevoir des recommandations de jeux Steam.

## Structure du Dossier

Conservez la structure du dossier et les noms des fichiers tels quels.

## Problèmes Connus et Résolution

- **Erreur de permission lors de l'exécution du script** : Assurez-vous que vous avez les droits nécessaires pour exécuter des scripts sur votre système. Vous pouvez accorder les permissions nécessaires en exécutant `chmod +x Recommender_Steam.sh`.

- **Python ou pip non trouvé** : Vérifiez que Python est correctement installé et que `python3` et `pip` sont accessibles depuis votre terminal. Si vous utilisez `pyenv` ou une autre gestion de versions Python, assurez-vous que la version correcte est activée.

Pour toute autre question ou problème, n'hésitez pas à contacter jonathan.ferreiro@skema.edu ou jules.vermelle@skema.edu
