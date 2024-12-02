# Steam Game Recommender

Welcome to the **Steam Game Recommender** repository! This project features a recommender system for video games, accessible via a Streamlit interface. It includes:

- Data files
- A Python script to run the recommender
- A demo video showcasing the functionality

Game information and reviews have been scraped from Steam, with preprocessing done to ensure clean data inputs. This project is a collaboration between a friend and myself.

## Features

- **Recommendation System**: Suggests games based on user preferences and data analysis.
- **Streamlit Interface**: Intuitive web interface for users to interact with the recommender.
- **Preprocessed Data**: Optimized and cleaned datasets for faster recommendations.

## Data & Demo

Due to size constraints, the data files and demo video are hosted externally. You can find:

- **Data Files**: Necessary inputs for the recommender system.
- **Demo Video**: A walkthrough of the system in action.

Access them via the link below:

[🔗 Google Drive - Data & Demo](https://drive.google.com/drive/folders/1kM9fziNo1yiWSwg-EgfdKQFK_tUFeokx?usp=sharing)

## Prerequisites

- macOS 10.15 (Catalina) or later.
- Python 3.8 or higher must be installed on your machine.
- Access to a terminal or command prompt.

## Installation

First you need to download all the data files from the google drive and store them into a "data" folder, then you could launch the application using the methods below

### Option 1: Running via the Bash Script

**Run the script**: Double-click the `Recommender_Steam.sh` file located in the application's main directory. The script will automatically:
    - Create a Python virtual environment if it doesn't already exist.
    - Install all required dependencies.
    - Configure necessary permissions.
    - Launch the Streamlit application in your default web browser.

### Option 2: Manual Execution

1. **Open a terminal**: Navigate to the `app_ressources` folder.

2. **Activate the virtual environment** (if already created):
    ```bash
    source venv_steam_recommender/bin/activate
    ```

3. **Launch the application**:
    ```bash
    streamlit run Steam_games_recommender.py
    ```

## Usage

Once the application is launched, either via the script or manually, your default web browser will open automatically with the Streamlit app loaded. Follow the on-screen instructions to start receiving Steam game recommendations.

## Folder Structure

Ensure the folder structure and file names are preserved as they are essential for the application to work correctly.

## Troubleshooting

- **Permission error when running the script**: Ensure you have the necessary permissions to execute scripts on your system. You can grant the required permissions by running:
    ```bash
    chmod +x Recommender_Steam.sh
    ```

- **Python or pip not found**: Verify that Python is correctly installed and that `python3` and `pip` are accessible from your terminal. If you use `pyenv` or another Python version manager, ensure the correct version is activated.

---