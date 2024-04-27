# NHL_Skater_Recommender_System
## Project Title: Finding Comparable NHL Skaters
This is a content-based recommender system. It takes NHL player data from individual performance totals and rates as well as on-ice performance totals and rates from 3 NHL regular seasons: 2021-22, 2022-23, 2023-24 that has been split into 4 different game states: all strengths (AS), even strength (ES), power play (PP) and penalty kill (PK) and based on the features in the selected game state outputs the 5 players most similar to the player in question.
The recommender uses a player’s index as its reference point and recommends the desired number of indices most similar to the index selected.
The unique feature of this recommender are currently twofold. First, since it is divided into different game-states, this recommender offers a more nuanced perspective of a player’s performance than those traditionally considered in contract arbitration such as regular season and playoffs. Secondly, it has the ability to perform comparisons outside of the season of the original input. For example, this recommender engine, using a player’s index from the 2021-2022 season can find similar indices of players in the 2023-24 season or other desired combinations. Something to note is that there are no records for goalies in this engine. A goalie-specific recommender system is something I hope to build in the future. 
## Project Motivation:
My motivation for this project begins with the simple fact that I am a big hockey fan. Beyond just being a hockey fan, I have always been into sports and have always been really into the amazing stats that are shown on the screen during games that offer such interesting nuances into the game within the game. I’ve always enjoyed the strategy behind building a roster, scouting, and trades. I aimed to build something that could track player development using comparable players in the league. 
I wanted to build something that was constantly improvable and scalable. This project is endlessly scalable because it can continue to include upcoming seasons as well as past seasons. 
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
## Usage:
### Season Notebooks:
The notebooks: 2021-2022 Season, 2022-2023 Season, and 2023-2024 Season contain the original data CSVs as well as the EDA and cleaning process involved in each season. The data was downloaded from NatturalStatrick.com. It uses the regular season data individual, and on-ice, counts and rates  as well as the bio data for the game-states all strengths, even strength, power play, and penalty kill.
### EDA:
The file “EDA_functions.py” is a Python script that contains all the functions used to clean the season data, combine the seasons into their respective game-state data frames and to build and access the players’ indices as well as the function to run the recommender engine itself.
## How to use the recommender:
The recommender uses the pairwise distances of a player’s index with reference to the game-state of a given season and returns the data for the 5 most similar players. The actual recommender is in the Jupyter Notebook called “All_Seasons.ipynb” which was also used to combine the game-states of each season into the data frames used in the recommender engine.
Steps to run the recommender:
### Step 1: Get the player’s index for the desired game state and season.
To do this, run the function: get_index_all_gamestates(‘player’s full name you want recommended’) which looks like this: Remember to adjust the parametrs for the desired game state. 
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
get_index_all_gamestates('Alex Ovechkin')
```
#### Output:
```python
Sidney Crosby's ALL STRENGTHS indices are: {2022: [28], 2023: [1020], 2024: [1963]}
Sidney Crosby's EVEN STRENGTH indices are: {2022: [28], 2023: [1020], 2024: [1963]}
Sidney Crosby's POWER PLAY indices are: {2022: [28], 2023: [932], 2024: [1794]}
Sidney Crosby's PENALTY KILL indices are: {2022: [28], 2023: [893], 2024: [1720]}

Alex Ovechkin's ALL STRENGTHS indices are: {2022: [18], 2023: [1013], 2024: [1959]}
Alex Ovechkin's EVEN STRENGTH indices are: {2022: [18], 2023: [1013], 2024: [1959]}
Alex Ovechkin's POWER PLAY indices are: {2022: [18], 2023: [925], 2024: [1790]}
Alex Ovechkin's PENALTY KILL indices are: {2022: [18], 2023: [886], 2024: [1716]}
```

### Step 2: *Optional*
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

### Step 3: Finding Similar NHL Players
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
#### Calling the function:
Using the appropriate function parameters would looks something like this:
```python
recommended_skaters_Nick_Suzuki = recommend_skaters(original_gamestate_df=df_all_stats_ES,
                                        processed_gamestate_df=df_all_stats_ES_encoded_dropped_scaled_pca,
                                        season=2022,
                                        player_index=749,
                                        top_n=6)
```
The sample output for the Montreal Canadiens' captain Nick Suzuki's 2022 stats for the EVEN STRENGTH game states are:

</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Player</th>
      <th>Team</th>
      <th>Position</th>
      <th>Age</th>
      <th>GP</th>
      <th>TOI</th>
      <th>Goals</th>
      <th>Total Assists</th>
      <th>First Assists</th>
      <th>Second Assists</th>
      <th>Total Points</th>
      <th>IPP</th>
      <th>Shots</th>
      <th>SH%</th>
      <th>ixG</th>
      <th>iCF</th>
      <th>iFF</th>
      <th>iSCF</th>
      <th>iHDCF</th>
      <th>Rush Attempts</th>
      <th>Rebounds Created</th>
      <th>PIM</th>
      <th>Total Penalties</th>
      <th>Minor</th>
      <th>Major</th>
      <th>Misconduct</th>
      <th>Penalties Drawn</th>
      <th>Giveaways</th>
      <th>Takeaways</th>
      <th>Hits</th>
      <th>Hits Taken</th>
      <th>Shots Blocked</th>
      <th>Faceoffs Won</th>
      <th>Faceoffs Lost</th>
      <th>Faceoffs %</th>
      <th>CF</th>
      <th>CA</th>
      <th>CF%</th>
      <th>FF</th>
      <th>FA</th>
      <th>FF%</th>
      <th>SF</th>
      <th>SA</th>
      <th>SF%</th>
      <th>GF</th>
      <th>GA</th>
      <th>GF%</th>
      <th>xGF</th>
      <th>xGA</th>
      <th>xGF%</th>
      <th>SCF</th>
      <th>SCA</th>
      <th>SCF%</th>
      <th>HDCF</th>
      <th>HDCA</th>
      <th>HDCF%</th>
      <th>HDGF</th>
      <th>HDGA</th>
      <th>HDGF%</th>
      <th>MDCF</th>
      <th>MDCA</th>
      <th>MDCF%</th>
      <th>MDGF</th>
      <th>MDGA</th>
      <th>MDGF%</th>
      <th>LDCF</th>
      <th>LDCA</th>
      <th>LDCF%</th>
      <th>LDGF</th>
      <th>LDGA</th>
      <th>LDGF%</th>
      <th>On-Ice SH%</th>
      <th>On-Ice SV%</th>
      <th>PDO</th>
      <th>Off. Zone Starts</th>
      <th>Neu. Zone Starts</th>
      <th>Def. Zone Starts</th>
      <th>On The Fly Starts</th>
      <th>Off. Zone Start %</th>
      <th>Off. Zone Faceoffs</th>
      <th>Neu. Zone Faceoffs</th>
      <th>Def. Zone Faceoffs</th>
      <th>Off. Zone Faceoff %</th>
      <th>TOI/GP</th>
      <th>Goals/60</th>
      <th>Total Assists/60</th>
      <th>First Assists/60</th>
      <th>Second Assists/60</th>
      <th>Total Points/60</th>
      <th>Shots/60</th>
      <th>ixG/60</th>
      <th>iCF/60</th>
      <th>iFF/60</th>
      <th>iSCF/60</th>
      <th>iHDCF/60</th>
      <th>Rush Attempts/60</th>
      <th>Rebounds Created/60</th>
      <th>PIM/60</th>
      <th>Total Penalties/60</th>
      <th>Minor/60</th>
      <th>Major/60</th>
      <th>Misconduct/60</th>
      <th>Penalties Drawn/60</th>
      <th>Giveaways/60</th>
      <th>Takeaways/60</th>
      <th>Hits/60</th>
      <th>Hits Taken/60</th>
      <th>Shots Blocked/60</th>
      <th>Faceoffs Won/60</th>
      <th>Faceoffs Lost/60</th>
      <th>CF/60</th>
      <th>CA/60</th>
      <th>FF/60</th>
      <th>FA/60</th>
      <th>SF/60</th>
      <th>SA/60</th>
      <th>GF/60</th>
      <th>GA/60</th>
      <th>xGF/60</th>
      <th>xGA/60</th>
      <th>SCF/60</th>
      <th>SCA/60</th>
      <th>HDCF/60</th>
      <th>HDCA/60</th>
      <th>HDGF/60</th>
      <th>HDGA/60</th>
      <th>MDCF/60</th>
      <th>MDCA/60</th>
      <th>MDGF/60</th>
      <th>MDGA/60</th>
      <th>LDCF/60</th>
      <th>LDCA/60</th>
      <th>LDGF/60</th>
      <th>LDGA/60</th>
      <th>Off. Zone Starts/60</th>
      <th>Neu. Zone Starts/60</th>
      <th>Def. Zone Starts/60</th>
      <th>On The Fly Starts/60</th>
      <th>Off. Zone Faceoffs/60</th>
      <th>Neu. Zone Faceoffs/60</th>
      <th>Def. Zone Faceoffs/60</th>
      <th>Season</th>
      <th>Age_Group</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>749</th>
      <td>Nick Suzuki</td>
      <td>MTL</td>
      <td>C</td>
      <td>23</td>
      <td>82</td>
      <td>1300.116667</td>
      <td>11</td>
      <td>26</td>
      <td>12</td>
      <td>14</td>
      <td>37</td>
      <td>64.91</td>
      <td>128</td>
      <td>8.59</td>
      <td>11.74</td>
      <td>202</td>
      <td>164</td>
      <td>121</td>
      <td>54</td>
      <td>9</td>
      <td>11</td>
      <td>22</td>
      <td>11</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>22</td>
      <td>62</td>
      <td>44</td>
      <td>84</td>
      <td>87</td>
      <td>48</td>
      <td>550</td>
      <td>548</td>
      <td>50.09</td>
      <td>1156</td>
      <td>1283</td>
      <td>47.40</td>
      <td>870</td>
      <td>957</td>
      <td>47.62</td>
      <td>646</td>
      <td>713</td>
      <td>47.53</td>
      <td>57</td>
      <td>83</td>
      <td>40.71</td>
      <td>51.01</td>
      <td>71.53</td>
      <td>41.63</td>
      <td>558</td>
      <td>694</td>
      <td>44.57</td>
      <td>203</td>
      <td>287</td>
      <td>41.43</td>
      <td>24</td>
      <td>52</td>
      <td>31.58</td>
      <td>355</td>
      <td>407</td>
      <td>46.59</td>
      <td>24</td>
      <td>17</td>
      <td>58.54</td>
      <td>515</td>
      <td>496</td>
      <td>50.94</td>
      <td>7</td>
      <td>8</td>
      <td>46.67</td>
      <td>8.82</td>
      <td>88.36</td>
      <td>0.972</td>
      <td>209</td>
      <td>270</td>
      <td>202</td>
      <td>891</td>
      <td>50.85</td>
      <td>413</td>
      <td>377</td>
      <td>429</td>
      <td>49.05</td>
      <td>15.855081</td>
      <td>0.51</td>
      <td>1.20</td>
      <td>0.55</td>
      <td>0.65</td>
      <td>1.71</td>
      <td>5.91</td>
      <td>0.54</td>
      <td>9.32</td>
      <td>7.57</td>
      <td>5.58</td>
      <td>2.49</td>
      <td>0.42</td>
      <td>0.51</td>
      <td>1.02</td>
      <td>0.51</td>
      <td>0.51</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>1.02</td>
      <td>2.86</td>
      <td>2.03</td>
      <td>3.88</td>
      <td>4.02</td>
      <td>2.22</td>
      <td>25.38</td>
      <td>25.29</td>
      <td>53.35</td>
      <td>59.21</td>
      <td>40.15</td>
      <td>44.17</td>
      <td>29.81</td>
      <td>32.90</td>
      <td>2.63</td>
      <td>3.83</td>
      <td>2.35</td>
      <td>3.30</td>
      <td>25.75</td>
      <td>32.03</td>
      <td>9.37</td>
      <td>13.24</td>
      <td>1.11</td>
      <td>2.40</td>
      <td>21.84</td>
      <td>25.04</td>
      <td>1.48</td>
      <td>1.05</td>
      <td>34.33</td>
      <td>33.06</td>
      <td>0.47</td>
      <td>0.53</td>
      <td>9.65</td>
      <td>12.46</td>
      <td>9.32</td>
      <td>41.12</td>
      <td>19.06</td>
      <td>17.40</td>
      <td>19.80</td>
      <td>2022</td>
      <td>Young Pro</td>
    </tr>
    <tr>
      <th>207</th>
      <td>Mikael Granlund</td>
      <td>NSH</td>
      <td>C</td>
      <td>30</td>
      <td>80</td>
      <td>1228.616667</td>
      <td>8</td>
      <td>28</td>
      <td>16</td>
      <td>12</td>
      <td>36</td>
      <td>55.38</td>
      <td>85</td>
      <td>9.41</td>
      <td>9.23</td>
      <td>162</td>
      <td>123</td>
      <td>107</td>
      <td>38</td>
      <td>5</td>
      <td>13</td>
      <td>27</td>
      <td>12</td>
      <td>11</td>
      <td>1</td>
      <td>0</td>
      <td>23</td>
      <td>38</td>
      <td>44</td>
      <td>89</td>
      <td>102</td>
      <td>31</td>
      <td>492</td>
      <td>533</td>
      <td>48.00</td>
      <td>1146</td>
      <td>1219</td>
      <td>48.46</td>
      <td>881</td>
      <td>921</td>
      <td>48.89</td>
      <td>614</td>
      <td>669</td>
      <td>47.86</td>
      <td>65</td>
      <td>71</td>
      <td>47.79</td>
      <td>56.71</td>
      <td>63.89</td>
      <td>47.02</td>
      <td>597</td>
      <td>560</td>
      <td>51.60</td>
      <td>213</td>
      <td>208</td>
      <td>50.59</td>
      <td>31</td>
      <td>35</td>
      <td>46.97</td>
      <td>384</td>
      <td>352</td>
      <td>52.17</td>
      <td>19</td>
      <td>16</td>
      <td>54.29</td>
      <td>501</td>
      <td>576</td>
      <td>46.52</td>
      <td>10</td>
      <td>12</td>
      <td>45.45</td>
      <td>10.59</td>
      <td>89.39</td>
      <td>1.000</td>
      <td>238</td>
      <td>243</td>
      <td>162</td>
      <td>873</td>
      <td>59.50</td>
      <td>502</td>
      <td>392</td>
      <td>359</td>
      <td>58.30</td>
      <td>15.357708</td>
      <td>0.39</td>
      <td>1.37</td>
      <td>0.78</td>
      <td>0.59</td>
      <td>1.76</td>
      <td>4.15</td>
      <td>0.45</td>
      <td>7.91</td>
      <td>6.01</td>
      <td>5.23</td>
      <td>1.86</td>
      <td>0.24</td>
      <td>0.63</td>
      <td>1.32</td>
      <td>0.59</td>
      <td>0.54</td>
      <td>0.05</td>
      <td>0.00</td>
      <td>1.12</td>
      <td>1.86</td>
      <td>2.15</td>
      <td>4.35</td>
      <td>4.98</td>
      <td>1.51</td>
      <td>24.03</td>
      <td>26.03</td>
      <td>55.97</td>
      <td>59.53</td>
      <td>43.02</td>
      <td>44.98</td>
      <td>29.98</td>
      <td>32.67</td>
      <td>3.17</td>
      <td>3.47</td>
      <td>2.77</td>
      <td>3.12</td>
      <td>29.15</td>
      <td>27.35</td>
      <td>10.40</td>
      <td>10.16</td>
      <td>1.51</td>
      <td>1.71</td>
      <td>25.00</td>
      <td>22.92</td>
      <td>1.24</td>
      <td>1.04</td>
      <td>35.34</td>
      <td>40.63</td>
      <td>0.71</td>
      <td>0.85</td>
      <td>11.62</td>
      <td>11.87</td>
      <td>7.91</td>
      <td>42.63</td>
      <td>24.52</td>
      <td>19.14</td>
      <td>17.53</td>
      <td>2022</td>
      <td>Prime Age</td>
    </tr>
    <tr>
      <th>183</th>
      <td>Charlie Coyle</td>
      <td>BOS</td>
      <td>C</td>
      <td>30</td>
      <td>82</td>
      <td>1131.250000</td>
      <td>13</td>
      <td>21</td>
      <td>13</td>
      <td>8</td>
      <td>34</td>
      <td>61.82</td>
      <td>105</td>
      <td>12.38</td>
      <td>12.31</td>
      <td>168</td>
      <td>148</td>
      <td>100</td>
      <td>46</td>
      <td>5</td>
      <td>20</td>
      <td>28</td>
      <td>10</td>
      <td>9</td>
      <td>0</td>
      <td>1</td>
      <td>13</td>
      <td>35</td>
      <td>30</td>
      <td>78</td>
      <td>67</td>
      <td>31</td>
      <td>469</td>
      <td>464</td>
      <td>50.27</td>
      <td>1091</td>
      <td>1053</td>
      <td>50.89</td>
      <td>881</td>
      <td>795</td>
      <td>52.57</td>
      <td>669</td>
      <td>571</td>
      <td>53.95</td>
      <td>55</td>
      <td>58</td>
      <td>48.67</td>
      <td>52.40</td>
      <td>51.62</td>
      <td>50.38</td>
      <td>512</td>
      <td>494</td>
      <td>50.89</td>
      <td>204</td>
      <td>200</td>
      <td>50.50</td>
      <td>26</td>
      <td>34</td>
      <td>43.33</td>
      <td>308</td>
      <td>294</td>
      <td>51.16</td>
      <td>19</td>
      <td>15</td>
      <td>55.88</td>
      <td>526</td>
      <td>508</td>
      <td>50.87</td>
      <td>10</td>
      <td>5</td>
      <td>66.67</td>
      <td>8.22</td>
      <td>89.84</td>
      <td>0.981</td>
      <td>154</td>
      <td>238</td>
      <td>165</td>
      <td>835</td>
      <td>48.28</td>
      <td>362</td>
      <td>351</td>
      <td>347</td>
      <td>51.06</td>
      <td>13.795732</td>
      <td>0.69</td>
      <td>1.11</td>
      <td>0.69</td>
      <td>0.42</td>
      <td>1.80</td>
      <td>5.57</td>
      <td>0.65</td>
      <td>8.91</td>
      <td>7.85</td>
      <td>5.30</td>
      <td>2.44</td>
      <td>0.27</td>
      <td>1.06</td>
      <td>1.49</td>
      <td>0.53</td>
      <td>0.48</td>
      <td>0.00</td>
      <td>0.05</td>
      <td>0.69</td>
      <td>1.86</td>
      <td>1.59</td>
      <td>4.14</td>
      <td>3.55</td>
      <td>1.64</td>
      <td>24.88</td>
      <td>24.61</td>
      <td>57.87</td>
      <td>55.85</td>
      <td>46.73</td>
      <td>42.17</td>
      <td>35.48</td>
      <td>30.29</td>
      <td>2.92</td>
      <td>3.08</td>
      <td>2.78</td>
      <td>2.74</td>
      <td>27.16</td>
      <td>26.20</td>
      <td>10.82</td>
      <td>10.61</td>
      <td>1.38</td>
      <td>1.80</td>
      <td>21.78</td>
      <td>20.79</td>
      <td>1.34</td>
      <td>1.06</td>
      <td>40.30</td>
      <td>38.92</td>
      <td>0.77</td>
      <td>0.38</td>
      <td>8.17</td>
      <td>12.62</td>
      <td>8.75</td>
      <td>44.29</td>
      <td>19.20</td>
      <td>18.62</td>
      <td>18.40</td>
      <td>2022</td>
      <td>Prime Age</td>
    </tr>
    <tr>
      <th>813</th>
      <td>Pius Suter</td>
      <td>DET</td>
      <td>C</td>
      <td>26</td>
      <td>82</td>
      <td>1132.483333</td>
      <td>14</td>
      <td>18</td>
      <td>12</td>
      <td>6</td>
      <td>32</td>
      <td>65.31</td>
      <td>137</td>
      <td>10.22</td>
      <td>16.85</td>
      <td>222</td>
      <td>178</td>
      <td>162</td>
      <td>71</td>
      <td>8</td>
      <td>21</td>
      <td>22</td>
      <td>11</td>
      <td>11</td>
      <td>0</td>
      <td>0</td>
      <td>9</td>
      <td>31</td>
      <td>29</td>
      <td>24</td>
      <td>34</td>
      <td>26</td>
      <td>431</td>
      <td>431</td>
      <td>50.00</td>
      <td>1004</td>
      <td>1121</td>
      <td>47.25</td>
      <td>751</td>
      <td>885</td>
      <td>45.90</td>
      <td>578</td>
      <td>664</td>
      <td>46.54</td>
      <td>49</td>
      <td>58</td>
      <td>45.79</td>
      <td>51.97</td>
      <td>54.99</td>
      <td>48.59</td>
      <td>523</td>
      <td>576</td>
      <td>47.59</td>
      <td>206</td>
      <td>239</td>
      <td>46.29</td>
      <td>22</td>
      <td>32</td>
      <td>40.74</td>
      <td>317</td>
      <td>337</td>
      <td>48.47</td>
      <td>17</td>
      <td>16</td>
      <td>51.52</td>
      <td>434</td>
      <td>510</td>
      <td>45.97</td>
      <td>7</td>
      <td>7</td>
      <td>50.00</td>
      <td>8.48</td>
      <td>91.27</td>
      <td>0.997</td>
      <td>164</td>
      <td>226</td>
      <td>156</td>
      <td>771</td>
      <td>51.25</td>
      <td>331</td>
      <td>308</td>
      <td>324</td>
      <td>50.53</td>
      <td>13.810772</td>
      <td>0.74</td>
      <td>0.95</td>
      <td>0.64</td>
      <td>0.32</td>
      <td>1.70</td>
      <td>7.26</td>
      <td>0.89</td>
      <td>11.76</td>
      <td>9.43</td>
      <td>8.58</td>
      <td>3.76</td>
      <td>0.42</td>
      <td>1.11</td>
      <td>1.17</td>
      <td>0.58</td>
      <td>0.58</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.48</td>
      <td>1.64</td>
      <td>1.54</td>
      <td>1.27</td>
      <td>1.80</td>
      <td>1.38</td>
      <td>22.83</td>
      <td>22.83</td>
      <td>53.19</td>
      <td>59.39</td>
      <td>39.79</td>
      <td>46.89</td>
      <td>30.62</td>
      <td>35.18</td>
      <td>2.60</td>
      <td>3.07</td>
      <td>2.75</td>
      <td>2.91</td>
      <td>27.71</td>
      <td>30.52</td>
      <td>10.91</td>
      <td>12.66</td>
      <td>1.17</td>
      <td>1.70</td>
      <td>22.39</td>
      <td>23.81</td>
      <td>1.20</td>
      <td>1.13</td>
      <td>33.21</td>
      <td>39.03</td>
      <td>0.54</td>
      <td>0.54</td>
      <td>8.69</td>
      <td>11.97</td>
      <td>8.27</td>
      <td>40.85</td>
      <td>17.54</td>
      <td>16.32</td>
      <td>17.17</td>
      <td>2022</td>
      <td>Prime Age</td>
    </tr>
    <tr>
      <th>264</th>
      <td>Mark Scheifele</td>
      <td>WPG</td>
      <td>C</td>
      <td>29</td>
      <td>67</td>
      <td>1174.316667</td>
      <td>22</td>
      <td>29</td>
      <td>16</td>
      <td>13</td>
      <td>51</td>
      <td>73.91</td>
      <td>122</td>
      <td>18.03</td>
      <td>13.85</td>
      <td>196</td>
      <td>159</td>
      <td>142</td>
      <td>66</td>
      <td>6</td>
      <td>13</td>
      <td>14</td>
      <td>7</td>
      <td>7</td>
      <td>0</td>
      <td>0</td>
      <td>19</td>
      <td>51</td>
      <td>47</td>
      <td>34</td>
      <td>108</td>
      <td>45</td>
      <td>336</td>
      <td>335</td>
      <td>50.07</td>
      <td>1179</td>
      <td>1176</td>
      <td>50.06</td>
      <td>898</td>
      <td>898</td>
      <td>50.00</td>
      <td>665</td>
      <td>643</td>
      <td>50.84</td>
      <td>69</td>
      <td>78</td>
      <td>46.94</td>
      <td>60.61</td>
      <td>70.07</td>
      <td>46.38</td>
      <td>633</td>
      <td>645</td>
      <td>49.53</td>
      <td>248</td>
      <td>291</td>
      <td>46.01</td>
      <td>31</td>
      <td>40</td>
      <td>43.66</td>
      <td>385</td>
      <td>354</td>
      <td>52.10</td>
      <td>22</td>
      <td>19</td>
      <td>53.66</td>
      <td>517</td>
      <td>471</td>
      <td>52.33</td>
      <td>13</td>
      <td>12</td>
      <td>52.00</td>
      <td>10.38</td>
      <td>87.87</td>
      <td>0.982</td>
      <td>194</td>
      <td>234</td>
      <td>123</td>
      <td>780</td>
      <td>61.20</td>
      <td>416</td>
      <td>348</td>
      <td>279</td>
      <td>59.86</td>
      <td>17.527114</td>
      <td>1.12</td>
      <td>1.48</td>
      <td>0.82</td>
      <td>0.66</td>
      <td>2.61</td>
      <td>6.23</td>
      <td>0.71</td>
      <td>10.01</td>
      <td>8.12</td>
      <td>7.26</td>
      <td>3.37</td>
      <td>0.31</td>
      <td>0.66</td>
      <td>0.72</td>
      <td>0.36</td>
      <td>0.36</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.97</td>
      <td>2.61</td>
      <td>2.40</td>
      <td>1.74</td>
      <td>5.52</td>
      <td>2.30</td>
      <td>17.17</td>
      <td>17.12</td>
      <td>60.24</td>
      <td>60.09</td>
      <td>45.88</td>
      <td>45.88</td>
      <td>33.98</td>
      <td>32.85</td>
      <td>3.53</td>
      <td>3.99</td>
      <td>3.10</td>
      <td>3.58</td>
      <td>32.34</td>
      <td>32.96</td>
      <td>12.67</td>
      <td>14.87</td>
      <td>1.58</td>
      <td>2.04</td>
      <td>26.23</td>
      <td>24.12</td>
      <td>1.50</td>
      <td>1.29</td>
      <td>38.16</td>
      <td>34.76</td>
      <td>0.96</td>
      <td>0.89</td>
      <td>9.91</td>
      <td>11.96</td>
      <td>6.28</td>
      <td>39.85</td>
      <td>21.25</td>
      <td>17.78</td>
      <td>14.26</td>
      <td>2022</td>
      <td>Prime Age</td>
    </tr>
    <tr>
      <th>419</th>
      <td>Alex Wennberg</td>
      <td>SEA</td>
      <td>C</td>
      <td>28</td>
      <td>80</td>
      <td>1190.100000</td>
      <td>9</td>
      <td>21</td>
      <td>13</td>
      <td>8</td>
      <td>30</td>
      <td>62.50</td>
      <td>73</td>
      <td>12.33</td>
      <td>6.47</td>
      <td>113</td>
      <td>89</td>
      <td>66</td>
      <td>38</td>
      <td>2</td>
      <td>9</td>
      <td>24</td>
      <td>12</td>
      <td>12</td>
      <td>0</td>
      <td>0</td>
      <td>10</td>
      <td>32</td>
      <td>40</td>
      <td>46</td>
      <td>79</td>
      <td>35</td>
      <td>463</td>
      <td>551</td>
      <td>45.66</td>
      <td>1057</td>
      <td>1024</td>
      <td>50.79</td>
      <td>783</td>
      <td>781</td>
      <td>50.06</td>
      <td>577</td>
      <td>571</td>
      <td>50.26</td>
      <td>48</td>
      <td>71</td>
      <td>40.34</td>
      <td>43.59</td>
      <td>58.50</td>
      <td>42.70</td>
      <td>499</td>
      <td>485</td>
      <td>50.71</td>
      <td>179</td>
      <td>205</td>
      <td>46.61</td>
      <td>23</td>
      <td>35</td>
      <td>39.66</td>
      <td>320</td>
      <td>280</td>
      <td>53.33</td>
      <td>14</td>
      <td>23</td>
      <td>37.84</td>
      <td>499</td>
      <td>463</td>
      <td>51.87</td>
      <td>10</td>
      <td>7</td>
      <td>58.82</td>
      <td>8.32</td>
      <td>87.57</td>
      <td>0.959</td>
      <td>213</td>
      <td>281</td>
      <td>193</td>
      <td>806</td>
      <td>52.46</td>
      <td>370</td>
      <td>374</td>
      <td>397</td>
      <td>48.24</td>
      <td>14.876250</td>
      <td>0.45</td>
      <td>1.06</td>
      <td>0.66</td>
      <td>0.40</td>
      <td>1.51</td>
      <td>3.68</td>
      <td>0.33</td>
      <td>5.70</td>
      <td>4.49</td>
      <td>3.33</td>
      <td>1.92</td>
      <td>0.10</td>
      <td>0.45</td>
      <td>1.21</td>
      <td>0.60</td>
      <td>0.60</td>
      <td>0.00</td>
      <td>0.00</td>
      <td>0.50</td>
      <td>1.61</td>
      <td>2.02</td>
      <td>2.32</td>
      <td>3.98</td>
      <td>1.76</td>
      <td>23.34</td>
      <td>27.78</td>
      <td>53.29</td>
      <td>51.63</td>
      <td>39.48</td>
      <td>39.37</td>
      <td>29.09</td>
      <td>28.79</td>
      <td>2.42</td>
      <td>3.58</td>
      <td>2.20</td>
      <td>2.95</td>
      <td>25.16</td>
      <td>24.45</td>
      <td>9.02</td>
      <td>10.34</td>
      <td>1.16</td>
      <td>1.76</td>
      <td>21.51</td>
      <td>18.82</td>
      <td>0.94</td>
      <td>1.55</td>
      <td>36.34</td>
      <td>33.72</td>
      <td>0.73</td>
      <td>0.51</td>
      <td>10.74</td>
      <td>14.17</td>
      <td>9.73</td>
      <td>40.64</td>
      <td>18.65</td>
      <td>18.86</td>
      <td>20.02</td>
      <td>2022</td>
      <td>Prime Age</td>
    </tr>
  </tbody>
</table>
</div>


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


























