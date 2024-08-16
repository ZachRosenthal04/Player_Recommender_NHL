# NHL_Skater_Recommender_System
## Project Title: Finding Comparable NHL Skaters
This is a content-based recommender system. It takes NHL player data from individual performance totals and rates as well as on-ice performance totals and rates from 3 NHL regular seasons: 2021-22, 2022-23, 2023-24 that has been split into 4 different game states: all strengths (AS), even strength (ES), power play (PP) and penalty kill (PK) and based on the features in the selected game state outputs the "n" most similar players. The recommender uses a player’s index as its reference point and uses the pairwise distances of indices when recommending.
The unique features of this recommender are currently twofold. First, since it is divided into different game states, this recommender offers a more nuanced perspective of a player’s performance than those traditionally considered in contract arbitration such as regular season and playoffs. Secondly, it can perform comparisons beyond the season of the original player index. For example, this recommender engine, using a player’s index from the 2021-2022 season can find similar indices of players in the 2023-24 season and so on. Something to note is that there are no records for goalies in this engine. A goalie-specific recommender system is something I hope to build in the future.
### New Data for Testing in branch:
I found another source for some data to use in the recommender from Money Puck. The data is slightly different than the Natural Stattrick sets and I a little more suer friendly. Additionally, the functionality of Money Puck's website makes using previous and future seasons much easier. My goal is to compare the recommendations using the same preprocessing to compare and comment on the recommendations. Additionally, I hope to do some statistical modeling with the data from Money Puck.
The MoneyPuck recommender is now complete. The recommendations are surprisingly similar and the most interesting part is that the MoneyPuck data doesn't include any information on age and yet the recommendations are quite similar. Using the all situations stats of Nick Suzuki (MTL), as an example, 
the Natural Stattrick recommender output the most similar skaters as: 1-Robert Thomas (STL), 2-Bo Horvat (NYI), 3-Nico Hischier (N.J), 4-Tim Stutzle (OTT), 5-Mika Zibanejad (NYR)
The MoneyPuck recommender for Nick Suzuki in all situations output:
1-Bo Horvat (NYI), 2-Robert Thomas (STL), 3-Dylan Larkin (DET), 4-Sean Monahan (WPG), 5-Mika Zibanejad (NYR)


## Project Motivation:
My motivation for this project began with the simple fact that I'm a big hockey fan and being from Montreal, hockey is always big news in the city.
Hockey isn't just big news, its big money. The NHL is currently valued at USD $42 billion! Not only this but the NHL has one of the strictest salary caps of the major North American professional sports leagues and it is currently valued at USD $83.5 million per team. Teams spend millions every year in player development, scouting, and contract negotiation to mitigate the salary cap and I believe this recommender can reduce the time and resources spent on these tasks, enabling a better and more effective use of teams' budgets and resources. This engine not only offers a tool that can recommend skaters teams may be interested in acquiring but also is a great tool for teams to better understand the value of the players they already have.

Beyond just being a hockey fan, I have always been into sports and have always been really into the amazing stats that are shown on the screen during games that offer such interesting nuances into the game within the game. I’ve always enjoyed the strategy behind building a roster, scouting, and trades. I aimed to build something that could track player development using comparable players in the league. 
I wanted to build something that was constantly improvable and scalable. This project is endlessly scalable because it can continue to include upcoming seasons as well as past seasons and with time can be adapted to other divisions or sports. 
## What problem or problems does this engine solve?
There are two problems I’m trying to solve with this recommender engine. The first problem is player performance tracking. For young players, it is extremely difficult to know how a player’s development is progressing because of how difficult it is to compare and analyze metrics beyond simple scoring metrics while also taking into account positional metrics’ nuances and different game states. This recommender offers a means of tracking player progression, decline, or consistency based on references of similar players. This leads me to the second problem I aim to solve – contract arbitration and navigating the NHL’s salary cap.
Salary arbitration is when a third party is needed to determine a fair salary for a player when a player (and their agent) cannot reach an agreement with the team’s offer. Most players opt for arbitration but few end up progressing to an arbitration hearing. That being said, the process is often financially and mentally taxing on the player and team and can sour or destroy the relationship between the players and their team. A key factor in contract negotiations and arbitration hearings is player comparisons which are used in salary benchmarking. This is similar to how house prices are determined by comparisons. This recommender offers a nuanced, data-driven comparison, available to both players and teams which can help teams manage the cap, plan for the future, or avoid lengthy arbitration hearings that can ruin important inter- and intra-team relationships.
## Installation:
The notebooks and recommender us the following libraries and packages:
```python
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import pairwise_distances
```
### EDA:
The file “EDA_functions.py” is a Python script that contains all the functions used to clean the season data, combine the seasons into their respective game-state data frames and to build and access the players’ indices as well as the function to run the recommender engine itself.
## How to use the recommender:
The recommender uses the pairwise distances of a player’s index with reference to the game state of a given season and returns the data for the 5 most similar players. The actual recommender is in the Jupyter Notebook called “All_Seasons.ipynb” which was also used to combine the game-states of each season into the data frames used in the recommender engine.
There are now two recommender engines. One is the recommend_skaters function which uses the Natural Stattrick data the other is MP_recommend_skaters function which uses data from MoneyPuck.com
Steps to run the recommender:
### Step 1-Natural Stattrick: Get the player’s index for the desired game state and season.
To do this, run the function: get_index_all_gamestates(‘player’s full name you want recommended’) which looks like this: Remember to adjust the parameters for the desired game state. 
```python
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
        f"{player_name}'s PENALTY KILL indices are: {PK_dict.get(player_name)}\n"
    )

    return print(result_string)
```
The expected output of this function will return a nested dictionary of players' indexes for each season in each of the game state data frames.
See these examples for Sidney Crosby and Alex Ovechkin below.
#### Use:
```python
get_index_all_gamestates('Sidney Crosby')
get_index_all_gamestates('Nick Suzuki')
```
#### Output:
```python
Sidney Crosby's ALL STRENGTHS indices are: {2022: [28], 2023: [1020], 2024: [1963]}
Sidney Crosby's EVEN STRENGTH indices are: {2022: [28], 2023: [1020], 2024: [1963]}
Sidney Crosby's POWER PLAY indices are: {2022: [28], 2023: [932], 2024: [1794]}
Sidney Crosby's PENALTY KILL indices are: {2022: [28], 2023: [893], 2024: [1720]}

Nick Suzuki's ALL STRENGTHS indices are: {2022: [749], 2023: [1643], 2024: [2508]}
Nick Suzuki's EVEN STRENGTH indices are: {2022: [749], 2023: [1643], 2024: [2508]}
Nick Suzuki's POWER PLAY indices are: {2022: [695], 2023: [1510], 2024: [2316]}
Nick Suzuki's PENALTY KILL indices are: {2022: [681], 2023: [1467], 2024: [2224]}
```
### Step 1-Money Puck: Get the player’s index for the desired game state and season.
To do this, run the function: MP_get_index_all_gamestates(‘player’s full name you want recommended’) which looks like this: Remember to adjust the parameters for the desired game state. 
def MP_get_index_all_gamestates(player_name, MP_AS_dict= MP_AS_player_dict, MP_5on5_dict= MP_5on5_player_dict, 
                                MP_4on5_dict= MP_4on5_player_dict, MP_5on4_dict= MP_5on4_player_dict,
                                MP_OS_dict= MP_OS_player_dict):
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
        f"{player_name}'s ALL SITUATIONS indices are: {MP_AS_dict.get(player_name)}\n"
        f"{player_name}'s 5-ON-5 indices are: {MP_5on5_dict.get(player_name)}\n"
        f"{player_name}'s 4-ON-5 indices are: {MP_4on5_dict.get(player_name)}\n"
        f"{player_name}'s 5-ON-4 indices are: {MP_5on4_dict.get(player_name)}\n"
        f"{player_name}'s OTHER SITUATIONS indices are: {MP_OS_dict.get(player_name)}\n"
    )

    return print(result_string)
The output will look very similar to the Natural Stattrickl version

### Step 2 - Natural Stattrick *Optional*
Get the baseline stats for the player and game state you want as the recommender's reference.
Run the function: get_players_baseline_gamestate_stats.
```python 
def get_players_baseline_gamestate_stats(original_gamestate_df, player_name):
    baseline_gamestate_stats = original_gamestate_df.loc[original_gamestate_df['Player'] == player_name]
    return baseline_gamestate_stats
```

The output of this function is the original game state data frame with all the player's stats for that game state.
The acceptable inputs for “original_gamestate_df” are: df_all_stats_AS, df_all_stats_ES, df_all_stats_PP, and df_all_stats_PK. 
If the player_name is misspelled or there is no data, the function returns an empty data frame.

### Step 2 - Natural Stattrick *Optional*
```python
def MP_get_players_baseline_gamestate_stats(original_gamestate_df, player_name):
    """
    Returns the baseline performance metrics of the player you are finding comparable players of 
    so you can see how their stats are over the course of the seasons in the engine.
    Args:
    - original_gamestate_df (pd.DataFrame): DataFrame containing the original skater stats.
    - player_name: must be a string of the full name of the player you want to look up, 
    If player name is misspelled or there is no data for that player, 
    the function returns an empty dataframe.
    -Small adustment from the other function. The MP function uses 'name' instead of 'Player' 

    """
    baseline_gamestate_stats = original_gamestate_df.loc[original_gamestate_df['name'] == player_name]
    return baseline_gamestate_stats
```
The output will look very similar to the Natural Stattrickl version

### Step 3- Natural Stattrick: Finding Similar NHL Players
Call the function: ```python recommend_skaters() ```
The function is:
```python
def recommend_skaters(original_gamestate_df, processed_gamestate_df, season, player_index, top_n=6):
    """
    Recommends skaters based on their stats using a preprocessed data frame. See All_Seasons Jupyter Notebook for preprocessing steps.

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

    distances = pairwise_distances(processed_gamestate_df, target_season_data)

    # Find the indices of the closest skaters
    indices = np.argsort(distances, axis=1)[:, :top_n]

    # Retrieve the recommended/similar players' stats from the original stats DataFrame
    recommended_skaters = original_gamestate_df[original_gamestate_df['Season'] == season].iloc[indices[player_index], :]
    return recommended_skaters
```
### Step 3- MoneyPuck: Finding Similar NHL Players
Call the function: ```python MP_recommend_skaters() ```
The function is:
```python
def MP_recommend_skaters(original_gamestate_df, processed_gamestate_df, season, player_index, top_n=6):
    """
    Recommends skaters based on their stats using a preprocessed PCA features.

    Args:
    - original_gamestate_df (pd.DataFrame): DataFrame containing the original skater stats.
        Acceptable inputs for original_gamestate_df are: [MP_AS_stats, MP_5on5_stats, MP_4on5_stats, MP_5on4_stats, MP_OS_stats]
    - processed_gamestate_df (pd.DataFrame): PCA-transformed and scaled features of the skaters.
        Acceptable inputs for processed_gamestate_df are: 
        [MP_AS_processed_data, MP_5on5_processed_data, MP_4on5_processed_data, MP_5on4_processed_data, MP_OS_processed_data]
    - season (int): The target season for comparison.
        Acceptable inputs for season are: 2021, 2022, 2023 
    - player_index (int): Index of the player in the DataFrame to get recommendations for.
        player_index as accessed through the function: MP_get_index_all_gamestates() 
    - top_n (int): Number of top recommendations to return.

    Returns:
    - pd.DataFrame: DataFrame containing the top_n recommended skaters for the given player in the specified season.
    """

    # Filter DataFrame for the target season
    target_season_data = processed_gamestate_df[original_gamestate_df['season'] == season]

    # Compute pairwise distances between all skaters and those from the target season
    distances = pairwise_distances(processed_gamestate_df, target_season_data)

    # Find the indices of the closest skaters
    indices = np.argsort(distances, axis=1)[:, :top_n]

    # Retrieve the recommendations from the original stats DataFrame
    MP_recommended_skaters = original_gamestate_df[original_gamestate_df['season'] == season].iloc[indices[player_index], :]

    return MP_recommended_skaters
```

## Contributing:
I encourage others to build on this project, as I will, by continuing to do feature engineering and using additional game state data as well as other seasons. Additionally, this project doesn’t take into account goaltender data and so this is one very clear space for additional contributions.
## Credits:
Though not technically involved in the direct creating of the project, I do want to shout out to Natural Stattrick.com which was my source for all the data used in this recommender.
## Future Considerations
### Short and medium term: 
I’m looking to do additional feature engineering including some feature engineering tailored to the specifics of each game-state and position. I would also like to build a complimentary recommender system using goalie data. Additionally, I’m hoping to have more features that indicate the players’ importance to their team to compliment and hopefully improve the quality of the recommendations.
### Long term: 
I’m hoping to be able to incorporate contract data and ML and DL algorithms to enable the recommender engine to be able to recommend similar players that also take into account contract and projected contract restrictions due to salary cap implications.  
## Contact Details:
Email: zach.rosenthal04@gmail.com
LinkedIn: zachary-rosenthal04/
GitHub: ZachRosenthal04


























