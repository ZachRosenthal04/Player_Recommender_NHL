#EDA_functions
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances

def calculate_playing_age(df, dob_col_name, age_col_name, season_year):
    """
    Updates the age of players in the DataFrame based on their date of birth.

    Parameters:
    df (pd.DataFrame): The DataFrame containing player data.
    dob_col_name (str): The name of the column with date of birth information.
    age_col_name (str): The name of the column where the age should be updated.
    current_year (int): The year to calculate current age from.

    Returns:
    pd.DataFrame: The DataFrame with updated ages.
    """
    # Extract the year and convert it to an integer
    df['Birth Year'] = df[dob_col_name].str[:4].astype(int)
    
    # Calculate the new age and replace the 'Age' column
    df[age_col_name] = season_year - df['Birth Year']
    
    # Drop the helper column
    df.drop(columns='Birth Year', inplace=True)
    
    return df


def index_sorter(df):
    df.sort_index()
    return df  # Return the modified dataframe



# Create a function for joining all the season and on-ice totals columns 
def concat_similar_dfs(df1, df2):
    columns_df1 = set(df1.columns)
    columns_df2 = set(df2.columns)
    shared_columns = columns_df1.intersection(columns_df2)
    df2_DROPPED_shared_columns = df2.drop(columns=shared_columns)
    return pd.concat([df1, df2_DROPPED_shared_columns], axis=1)


# List of columns to exclude from renaming
excluded_columns = ['Player', 'Team', 'Position', 'Age', 'Height (in)', 'Weight (lbs)']
def game_phase_col_renamer(df_list, excluded_col, str_to_add):
    """
    Rename columns in a list of pandas DataFrames, prepending a specified string to each column name
    except for those listed in the excluded columns. The string is added only if it is not already present
    at the beginning of the column name.

    Parameters:
    - df_list (list of pd.DataFrame): A list of pandas DataFrame objects whose column names are to be modified.
    - excluded_col (list of str): Column names that should not be modified.
    - str_to_add (str): The string to prepend to column names that are not in the excluded list and do not 
      already start with this string.

    Returns:
    - list of pd.DataFrame: The list of DataFrames with modified column names. Note that the renaming is 
      done in-place; this function returns the modified list as a convenience."""
    excluded_columns= excluded_col
    for df in df_list:
        df.columns = [str_to_add + col if not col.startswith(str_to_add) and col not in excluded_col else col for col in df.columns]
    return df_list

# FROM: All_Seasons.ipynb - CREATE PLAYER INDECES NESTED DICTIONARIES
def create_player_index_dict(df):
    
    """Create a nested dictionary from a DataFrame that maps player names to their indices for each season.

    This function resets the index of the DataFrame to ensure that the index column 
    holds the original row indices. It then groups the DataFrame by 'Player' and 'Season' 
    and aggregates the indices into a list for each group. After grouping, it pivots the DataFrame 
    so each 'Player' name is a row with each 'Season' as columns, containing lists of indices 
    as values. Finally, it converts the pivoted DataFrame into a nested dictionary where each player's 
    name is a key to a dictionary mapping each season to the player's indices.

    Parameters:
    df (pandas.DataFrame): The DataFrame to process, which must contain 'Player' and 'Season' columns 
                           and has a unique index.

    Returns:
    dict: A nested dictionary where the first level keys are player names, and second level keys are 
          seasons, each mapping to a list of index positions for that player in that season.
    """

    # Reset the index 
    df = df.reset_index()

    # Group by 'Player' and 'Season', then aggregate the original index values into a list.
    grouped = df.groupby(['Player', 'Season'])['index'].agg(lambda x: list(x)).reset_index()

    # Pivot the DataFrame to have 'Player' as rows and 'Season' as columns with list of indices as values.
    pivot_df = grouped.pivot(index='Player', columns='Season', values='index')

    # Convert the pivoted DataFrame into a nested dictionary.
    player_index_dict = pivot_df.apply(lambda row: row.dropna().to_dict(), axis=1).to_dict()

    return player_index_dict

def get_index_all_gamestates(player_name, AS_dict= player_index_dict_AS, ES_dict= player_index_dict_ES, PP_dict= player_index_dict_PP, PK_dict= player_index_dict_PK):
    """
    Returns a string with all the indices for each game state (All Strengths, Even Strength,
    Power Play, and Penalty Kill) for a given player.
    
    Parameters:
    - player_name (str): The name of the player to lookup.
    - player_index_dict_AS (dict): The dictionary with indices for All Strengths.
    - player_index_dict_ES (dict): The dictionary with indices for Even Strength.
    - player_index_dict_PP (dict): The dictionary with indices for Power Play.
    - player_index_dict_PK (dict): The dictionary with indices for Penalty Kill.

    Returns:
    - str: A formatted string containing the indices for each game state for the player.
    """
    result_string= (
        f"{player_name}'s ALL STRENGTHS indices are: {AS_dict.get(player_name)}\n"
        f"{player_name}'s EVEN STRENGTH indices are: {ES_dict.get(player_name)}\n"
        f"{player_name}'s POWER PLAY indices are: {PP_dict.get(player_name)}\n"
        f"{player_name}'s PENALTY KILL indices are: {PK_dict.get(player_name)}"
    )

    return print(result_string)

def get_players_baseline_gamestate_stats(original_gamestate_df, player_name):
    """
    Returns the baseline performance metrics of the player you are finding comparable players of 
    so you can see how their stats are over the course of the seasons in the engine.
    Args:
    - original_gamestate_df (pd.DataFrame): DataFrame containing the original skater stats.
    - player_name: must be a string of the full name of the player you want to look up, 
    If player name is misspelled or there is no data for that player, 
    the function returns an empty dataframe 

    """
    baseline_gamestate_stats = original_gamestate_df.loc[original_gamestate_df['Player'] == player_name]
    return baseline_gamestate_stats

def recommend_skaters(original_gamestate_df, processed_gamestate_df, season, player_index, top_n=6):
    """
    Recommends skaters based on their stats using a preprocessed PCA features.

    Args:
    - original_gamestate_df (pd.DataFrame): DataFrame containing the original skater stats.
        Acceptable inputs for original_gamestate_df are: [df_all_stats_AS, df_all_stats_ES, df_all_stats_PP, df_all_stats_PK]
    - processed_gamestate_df (pd.DataFrame): PCA-transformed and scaled features of the skaters.
        Acceptable inputs for processed_gamestate_df are: 
        [df_all_stats_AS_encoded_dropped_scaled_pca, df_all_stats_ES_encoded_dropped_scaled_pca, 
        df_all_stats_PP_encoded_dropped_scaled_pca, df_all_stats_PK_encoded_dropped_scaled_pca]
    - season (int): The target season for comparison.
        Acceptable inputs for season are: 2022, 2023, 2024
    - player_index (int): Index of the player in the DataFrame to get recommendations for.
        player_index as accessed through the function: get_index_all_gamestates() 
    - top_n (int): Number of top recommendations to return.

    Returns:
    - pd.DataFrame: DataFrame containing the top_n recommended skaters for the given player in the specified season.
    """

    # Filter DataFrame for the target season
    target_season_data = processed_gamestate_df[original_gamestate_df['Season'] == season]
    # Compute pairwise distances between all skaters and those from the target season
    distances = pairwise_distances(processed_gamestate_df, target_season_data)
    # Find the indices of the closest skaters
    indices = np.argsort(distances, axis=1)[:, :top_n]
    # Retrieve the recommendations from the original stats DataFrame
    recommended_skaters = original_gamestate_df[original_gamestate_df['Season'] == season].iloc[indices[player_index], :]

    return recommended_skaters


def made_playoffs(row):
    ''' made_playoffs functon creates the 'Playoff_Team' column in each of the game states dataframes using the .apply() method.
    Future additions would just need to create a variable for the list of playoff teams and then and a new 'elif' line for the most recent season.'''
    if row['Season'] == 2022 and row['Team'] in playoff_teams_2022:
        return 1
    elif row['Season'] == 2023 and row['Team'] in playoff_teams_2023:
        return 1
    elif row['Season'] == 2024 and row['Team'] in playoff_teams_2024:
        return 1
    else:
        return 0


