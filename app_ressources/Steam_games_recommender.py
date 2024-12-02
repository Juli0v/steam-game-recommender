# Pour lancer l'application depuis VS Code :
# Clique droit sur le fichier -> "Open in integrated terminal" 
# Run: streamlit run Steam_games_recommender.py

# Importation des librairies :
import pandas as pd
import numpy as np
import pickle
import streamlit as st
from scipy.sparse import coo_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import linear_kernel
import urllib.request

#--------------

# Initialisation de la largeur de la page Streamlit
st.set_page_config(layout="wide")

#--------------

# Chargement des df précédemment créer pour éviter leur création à chaque lancement de l'app
# Définition d'une fonction pour pouvoir la décorer de la fonction de mise en cache de streamlit
@st.cache_data
def load_data():
    #init_df est le df contenant les jeux. C'est le csv initiale qui a été preprocess dans BigQuery pour garder les jeux qui avaient entre 5 et 50 reviews par user unique
    init_df = pd.read_csv('Data/games_reduced.csv')
    #df est le df contenant les reviews
    df = pd.read_csv('Data/review_with_nlp_processed.csv')
    return init_df, df

init_df, df = load_data()

# Charger les résultats de Tokenizations précédemment effectuée et sauvegardée
with open('Data/tokenization_cosine_sim.pkl', 'rb') as f:
    cosine_sim = pickle.load(f)

with open('Data/tokenization_indices.pkl', 'rb') as f:
    indices = pickle.load(f)

#--------------

# Afin de faciliter la visualisation lors du dev, le init_df est scindé en plusieurs df selon les usages :
@st.cache_data
def process_df(init_df): 
    df_comput = init_df.dropna(subset=['Name'])

    #We are combining all the texts used to describe the games, such as the 'tags', 'categories', 'genres' and the description of the game. We will tokenize it later
    df_comput['combined_text'] = df_comput['Categories'] + ' ' + df_comput['Genres'] + ' ' + df_comput['Tags'] + ' ' + df_comput['About_the_game']
    df_comput['combined_text'] = df_comput['combined_text'].fillna('')

    # df_categories is to keep the filter feature for the end of the recommender
    df_categories = init_df.drop(['Peak_CCU','DLC_count','Reviews','Metacritic_url','Metacritic_score','Header_image','Website','Support_url','Support_email', 'Screenshots', 'Movies', 'Notes','Full_audio_languages', 'Developers','Publishers','About_the_game', 'Categories','Tags', 'User_score', 'Positive', 'Negative', 'Score_rank','Achievements','Recommendations','Average_playtime_forever', 'Average_playtime_two_weeks', 'Median_playtime_forever', 'Median_playtime_two_weeks' ], axis=1)
    df_categories = df_categories.dropna(subset=['Name'])

    # df_score is to keep the score to apply weights for the end of the recommender
    df_score = init_df.drop(['Peak_CCU','DLC_count','Reviews','Metacritic_url','Metacritic_score','Header_image','Website','Support_url','Support_email', 'Screenshots', 'Movies', 'Notes','Full_audio_languages', 'Developers','Publishers','Estimated_owners', 'Required_age', 'Price','Release_date','Supported_languages', 'Windows', 'Mac', 'Linux', 'User_score', 'Score_rank', 'Achievements', 'Average_playtime_forever', 'Average_playtime_two_weeks', 'Median_playtime_forever', 'Median_playtime_two_weeks','About_the_game', 'Categories', 'Genres', 'Tags'  ], axis=1)
    df_score = df_score.dropna(subset=['Name'])

    df_comput = df_comput.reset_index(drop=True)
    df_categories = df_categories.reset_index(drop=True)
    df_score = df_score.reset_index(drop=True)

    return df_comput, df_categories, df_score

df_comput, df_categories, df_score = process_df(init_df)

#--------------

# Pour la partie : User_based
# Création de la matrice creuse (Beaucoup plus efficace et moins gournmande que la première méthode avec pivot table + svd)

data = df['rating'].values
row = df['user_id'].values
col = df['product_id'].values
max_row = df['user_id'].max() + 1
max_col = df['product_id'].max() + 1

sparse_matrix = coo_matrix((data, (row, col)), shape=(max_row, max_col))

user_ids = np.arange(max_row)
product_ids = np.arange(max_col)


# Fonction qui prend en input la matrice creuse créée au-dessus, la liste des jeux aimés par l'utilisateur du recommender, et la liste de tous les user ids
# Renvoie liste de tuple contenant (id des users avec le plus de jeux aimés en commun, nombre de jeux aimés en commun)

def find_best_user_sparse(sparse_matrix, game_ids, user_ids):
    # Conversion au format CSR pour améliorer l'efficacité des opérations sur les lignes
    csr = sparse_matrix.tocsr()
    
    game_ids = init_df[init_df['Name'].isin(game_ids)]['AppID'].unique()
    
    # Filtre sur les game_ids
    filtered_csr = csr[:, game_ids]
    
    # Calcul de la moyenne des notes pour chaque user pour les jeux sélectionnés 
    # (Pour valoriser les joueurs ayant des jeux aimés en commun, la moyenne est majorée du nombre de jeux aimés en commun )
    ratings_sum = filtered_csr.sum(axis=1).A1
    games_rated_count = (filtered_csr != 0).sum(axis=1).A1
    high_ratings_count = (filtered_csr >= 4).sum(axis=1).A1
    user_means = (ratings_sum / games_rated_count)+ high_ratings_count
    
    # Gestion des division par zéro pour les users qui n'ont pas de notes pour certains des jeux sélectionnés
    user_means[np.isnan(user_means)] = 0
    
    # Récupération des indices des top 10 users basé sur la moyenne ajustée 
    top_user_indices = np.argsort(-user_means)[:10]
    
    # Préparation des résultats pour récupérer les indices des users + leur nombres de jeux aimés en commun
    top_users_info = [(user_ids[idx], high_ratings_count[idx]) for idx in top_user_indices]
    
    return top_users_info

# Une fois les top users similaires identifiés, nous allons récupérer leurs jeux + leurs notes
# La fonctions prends en input : la matrice creuse, les ids des top users similaire, la liste des product id
# Elle renvoie un df avec les AppID et leur score associé 
def extract_user_ratings_sparse_2(sparse_matrix, user_ids, product_ids):
    all_results = []  # Liste pour stocker les résultats intermédiaires
    
    # Assurer que sparse_matrix est en format CSR pour les opérations efficaces sur les lignes
    csr = sparse_matrix.tocsr()
    
    for user_id in user_ids:  # Itérer sur chaque user_id des tops users
        # Extraire la ligne correspondant au user_id
        user_ratings_row = csr[user_id[0], :]
        
        # Convertir en format dense pour faciliter la manipulation et le filtrage
        user_ratings_dense = user_ratings_row.toarray().ravel()
        
        # Filtrer les notes > 3 et obtenir les product_ids des jeux appréciés
        positive_ratings_indices = user_ratings_dense > 3
        positive_product_ids = product_ids[positive_ratings_indices]
        positive_ratings = user_ratings_dense[positive_ratings_indices]

        # Bonifier le score en fonction du nombre de notes positives (plus un user à de jeux en commun avec l'utilisateur, plus ses jeux doivent être valorisés par le recommender)
        # On augmente le score de chaque produit par une fraction du nombre de notes positives
        bonus_factor = 0.05 * user_id[1] 
        positive_ratings_adjusted = positive_ratings + bonus_factor
        
        # Créer un DataFrame pour les résultats de cet user_id
        result_df = pd.DataFrame({'AppID': positive_product_ids, 'Score': positive_ratings_adjusted})
        
        # Trier le DataFrame par 'Score' en ordre décroissant
        result_df.sort_values(by='Score', ascending=False, inplace=True)
        
        # Ajouter le DataFrame intermédiaire à la liste
        all_results.append(result_df)
    
    # Concaténer tous les DataFrames intermédiaires de chaque user en un seul DataFrame final
    final_df = pd.concat(all_results).reset_index(drop=True)

    # On supprime les jeux en doublons en gardant la ligne dont le score est le plus haut
    final_df = final_df.sort_values(by='Score', ascending=False).drop_duplicates(subset='AppID', keep='first')

    return final_df


#--------------
# Pour la partie : Item_based
# L'étape de Tokenization nous a permis de créer notre matrice de cosimilarité entre tousl les jeux

# Cette fonction prend en input les jeux que l'utilisateurs aiment + la matrice de cosimilarité
# Elle renvoie un df contenant les jeux similaires
def get_recommendations(selected_games, cosine_sim=cosine_sim):
    all_sim_scores = [] # Liste pour stocker les résultats intermédiaires

    # Collecter les scores de similarité pour chaque jeu sélectionné par l'usager
    for game in selected_games:
        if game in indices:  # Vérifier si le jeu existe dans les indices
            idx = indices[game]
            sim_scores = list(enumerate(cosine_sim[idx]))
            # Ajouter un identifiant de jeu pour distinguer les jeux sélectionnés
            sim_scores = [(i, score, game) for i, score in sim_scores]
            all_sim_scores.extend(sim_scores)
        else:
            st.write(f"Le jeu sélectionné '{game}' n'est pas trouvé dans les indices.")

    # Trier tous les scores de similarité
    all_sim_scores = sorted(all_sim_scores, key=lambda x: x[1], reverse=True)
    recommended_indices = []
    scores = []
    for i, score, game in all_sim_scores:
        if i not in recommended_indices:
            recommended_indices.append(i)
            scores.append(score)

    df_comput['Release_date'] = pd.to_datetime(df_comput['Release_date'], errors='coerce')

    # Filtrer les recommandations basées sur les critères spécifiés
    filtered_recommendations = df_comput.loc[recommended_indices].copy()

    # Nous transformons notre de score de cosimilarity en score sur 5 pour correspondre à la notation de la partie user_based.
    note_final_max = 5
    note_actual_max = 0.5

    # Ajuster chaque score individuellement et stocker les résultats dans une nouvelle liste
    score_ajusted = [min(score / note_actual_max * note_final_max, note_final_max) for score in scores]


    filtered_recommendations['Similarité'] = score_ajusted

    return filtered_recommendations


#Note : par choix, la partie items_based renvoie les meilleurs similarity par jeux et non par rapport au mix de jeux

#--------------

# Rassemblement des recommenders

# Cette fonction sert à rassembler recommandations user et items based
# Elle prend en input : les recommandations items, les recommandations users, le df initial, et la variable orienation définie par l'utilisateur qui permet d'orienter le recommender entre l'approche Item ou User based. 

def mix_recommenders(items_df, users_df, initial_df, orientation):

    # Assembler les AppID de items_df et users_df, en excluant les doublons
    unique_appids = pd.concat([items_df['AppID'], users_df['AppID']]).drop_duplicates().tolist()

    # Première fusion : initial_df avec items_df pour obtenir "Similarité"
    merged_df_1 = initial_df.merge(items_df[['AppID', 'Similarité']], on='AppID', how='outer')
    
    # Deuxième fusion : Le résultat de la première fusion avec users_df pour obtenir "Score"
    # La clé commune est "AppID"
    final_merged_df = merged_df_1.merge(users_df[['AppID', 'Score']], on='AppID', how='outer')
    final_merged_df['Score'].fillna(0, inplace=True)

    # Ajouter "Similarité Ajustée" et "Score Ajusté" en utilisant 'orientation' comme coefficient
    final_merged_df['Similarité Ajustée'] = final_merged_df['Similarité'] * (1 - orientation/100)
    final_merged_df['Score Ajusté'] = final_merged_df['Score'] * orientation/100

    # Le score finale est la somme de la similarité ajustée et des score des users ajusté
    final_merged_df['Score Final'] = final_merged_df['Similarité Ajustée']+final_merged_df['Score Ajusté']
        
    # Filtrer pour ne garder que les jeux dont l'AppID est dans l'ensemble unique des AppID
    final_merged_df = final_merged_df[final_merged_df['AppID'].isin(unique_appids)]
    
    # Sélections des colonnes à garder et filtre par score final
    result = final_merged_df[['AppID','Name','Similarité', 'Score','Similarité Ajustée', 'Score Ajusté','Score Final','Price','Release_date','Required_age','Windows','Linux','Mac']]
    result = result.sort_values(by='Score Final', ascending=False)
    
    return result

#--------------

# Fonction pour vérifier la connectivité internet pour l'affichage des images et des liens des jeux 
def check_internet():
    try:
        # Tente d'ouvrir une connexion à un site commun (ici, Google)
        urllib.request.urlopen('http://www.google.com/', timeout=1)
        return True
    except urllib.error.URLError as Error:
        return False

#--------------

# Applications des filtres choisits par l'utilisateur

# Fonction qui prend les recommandations finales et les filtres en input

def filtering_recommendations(df_to_filter,age_max, max_price, year_mini, year_max, windows, linux, mac):

    df_to_filter['Release_date'] = pd.to_datetime(df_to_filter['Release_date'], errors='coerce')


    # Applications des filtres:
    filtered_recommendations = df_to_filter[
        (df_to_filter['Price'] <= max_price) &
        (df_to_filter['Release_date'].dt.year >= year_mini) &
        (df_to_filter['Release_date'].dt.year <= year_max) &
        (df_to_filter['Required_age'] <= age_max)
    ]

    # Autre système de filtre pour filtrer les OS :
    conditions = []
    if 'Windows' in filtered_recommendations.columns and windows:
        conditions.append(filtered_recommendations['Windows'] == True)
    if 'Linux' in filtered_recommendations.columns and linux:
        conditions.append(filtered_recommendations['Linux'] == True)
    if 'Mac' in filtered_recommendations.columns and mac:
        conditions.append(filtered_recommendations['Mac'] == True)

    # Combine les conditions s'il y en a, sinon utilise une condition toujours vraie
    if conditions:
        os_conditions = np.logical_or.reduce(conditions)
    else:
        os_conditions = pd.Series([True] * len(filtered_recommendations))

    filtered_recommendations = filtered_recommendations[os_conditions]

    return filtered_recommendations


#--------------

# Interface Streamlit


st.title('Steam Games Recommender')


# Création de colonnes pour ajuster des éléments par la suite:
col_jeux, col_filtres = st.columns([3, 2])
col_gauche, col_centre, col_droite = st.columns([3,2,2])
col_gauche_petite, col_centre_grande, col_droite_petite = st.columns([3,5,2])

# État d'affichage du df final
afficher_df = False


with col_jeux:
    st.subheader('Jeux :')

    # Récupération de la liste de tous les jeux disponible dans le recommender.
    jeux = df_comput['Name'].unique()
    # Input à choix multiple pour indiquer les jeux appréciés par l'utilisateur
    selected_games = st.multiselect("Quels jeux aimes-tu ?", jeux)

    # Création d'un slider pour définir l'orientation item ou user based du recommender.
    orientation = st.slider(
        'Choisissez votre type de recommandation : Orienté Jeux -> Orienté Joueurs',min_value=0, max_value=99, value = 40
    )

    # Input numérique pour choisir le nombre de recommandations dans le tableau final
    nb_reco = st.number_input(
        "Choisissez le nombre de recommandations souhaité :",min_value=3, max_value=100, value=10, step=1
    )


with col_filtres:
    st.subheader('Filtres :')

    # Input pour les différents filtres :
    age_max = st.slider('PEGI (ans)', min_value = 0, max_value=18, value =18)
    max_price = st.slider('Prix maximum (€)', min_value = 0, max_value=100, value = 100)

    year_mini = st.number_input('Année mini : ', min_value=1980, max_value=2023, value = 2000, step=1)
    year_max = st.number_input('Année max : ', min_value=1981, max_value=2024, value = 2024, step=1)

    st.text('OS supporté : ')
    windows = st.checkbox("Windows", value = True)
    linux = st.checkbox("Linux")
    mac = st.checkbox("Mac")


with col_centre:
    st.write("")
    st.write("")

    # Boutton pour lancer la recommandation !
    if st.button('Recommander') and selected_games:
        items_df = get_recommendations(selected_games, cosine_sim=cosine_sim)
        best_user_id = find_best_user_sparse(sparse_matrix, selected_games, user_ids)
        users_df = extract_user_ratings_sparse_2(sparse_matrix, best_user_id, product_ids)
        mixed_df = mix_recommenders(items_df, users_df, df_categories, orientation)
        result_df = filtering_recommendations(mixed_df,age_max, max_price, year_mini, year_max, windows, linux, mac)

        # Mise à jour de la variable basée sur l'état du DataFrame
        afficher_df = not result_df.empty
        

with col_centre_grande:
    if afficher_df:
        if not result_df.empty:
            # Retirer les jeux sélectionnés par l'user des résultats du recommender.
            result_df_filtered = result_df[~result_df['Name'].isin(selected_games)]
            #Sélectionner les colonnes spécifiques à afficher
            columns_to_display = ['Name','Similarité','Similarité Ajustée','Score','Score Ajusté','Score Final','Price','Release_date','Required_age','Windows','Linux','Mac','AppID']
            #Limiter le nombre de lignes à celui indiqué et afficher les colonnes spécifiées
            st.dataframe(result_df_filtered[columns_to_display].head(nb_reco-1))

        else:
            st.write("Aucun jeu ne correspond à ta recherche. Essaie de modifier tes filtres.")


if afficher_df:
    if not result_df.empty:
        if check_internet():
            # Votre code qui dépend de l'internet ici
            games = result_df_filtered[['AppID', 'Name']].head(nb_reco-1)
            nb_cols = 3
            rows = (len(games) + nb_cols - 1) // nb_cols
            for i in range(rows):
                cols = st.columns(nb_cols)
                for col_num in range(nb_cols):
                    index = i*nb_cols + col_num
                    if index < len(games):
                        app_id, game_name = games.iloc[index]
                        image_url = f"https://cdn.cloudflare.steamstatic.com/steam/apps/{app_id}/header.jpg"
                        steam_url = f"https://store.steampowered.com/app/{app_id}"
                        with cols[col_num]:
                            st.image(image_url)
                            st.markdown(f"<div style='text-align: center'><a href='{steam_url}' target='_blank'>{game_name}</a></div>", unsafe_allow_html=True)
        else:
            # Alternative ou message d'erreur si pas de connexion internet
            st.error("Vous avez besoin d'une connexion internet pour accéder aux images et aux pages des jeux. Veuillez vérifier votre connexion.")


#--------------