# NHL_Skater_Recommender_System
## Project Title: Finding Comparable MHL Skaters
This is a content-based recommender system. It takes NHL player data from individual performance totals and rates as well as on-ice performance totals and rates from 3 NHL regular seasons: 2021-22, 2022-23, 2023-24 that has been split into 4 different game states: all strengths (AS), even strength (ES), power play (PP) and penalty kill (PK) and based on the features in the selected game state outputs the 5 players most similar to the player in question.
The recommender uses a player’s index as its reference point and recommends the desired number of indices most similar to the index selected.
The unique feature of this recommender are currently twofold. First, since it is divided into different game-states, this recommender offers a more nuanced perspective of a player’s performance than those traditionally considered in contract arbitration such as regular season and playoffs. Secondly, it has the ability to perform comparisons outside of the season of the original input. For example, this recommender engine, using a player’s index from the 2021-2022 season can find similar indices of players in the 2023-24 season or other desired combinations. Something to note is that there are no records for goalies in this engine. A goalie-specific recommender system is something I hope to build in the future. 
## Project Motivation:
My motivation for this project begins with the simple fact that I am a big hockey fan. Beyond just being a hockey fan, I have always been into sports and have always been really into the amazing stats that are shown on the screen during games that offer such interesting nuances into the game within the game. I’ve always enjoyed the strategy behind building a roster, scouting, and trades. I aimed to build something that could track player development using comparable players in the league. 
I wanted to build something that was constantly improvable and scalable. This project is endlessly scalable because it can continue to include upcoming seasons as well as past seasons. 
What problem are you trying to solve?
There are two problems I’m trying to solve with this recommender engine. The first problem is player performance tracking. For young players, it is extremely difficult to know how a player’s development is progressing because of how difficult it is to compare and analyze metrics beyond simple scoring metrics while also taking into account positional metrics’ nuances and different game states. This recommender offers a means of tracking player progression, decline, or consistency based on referencing of similar players. Which leads me to the second problem I aim to solve – contract arbitration and navigating the NHL’s salary cap.
Salary arbitration is when a third party is needed to determine a fair salary for a player when a player (and their agent) cannot come to an agreement with the team’s offer. Most players opt for arbitration but few end up progressing to an arbitration hearing. That being said, the process is often financially and mentally taxing on the player and team and can sour or destroy the relationship between the players and their team. An key factor in contract negotiations and arbitration hearings are player comparisons which are used in salary benchmarking. This is similar to how house prices are determined by comparisons. This recommender offers a nuanced, data-driven comparison, available to both players and teams which can help teams manage the cap, plan for the future, or avoid lengthy arbitration hearings that can ruin important relationships.
## Installation:
## Usage:
### Season Notebooks:
The notebooks: 2021-2022 Season, 2022-2023 Season, and 2023-2024 Season contain the original data CSVs as well as the EDA and cleaning process involved in each season. The data was downloaded from NatturalStatrick.com. It uses the regular season data individual, and on-ice, counts and rates  as well as the bio data for the game-states all strengths, even strength, power play, and penalty kill.
### EDA:
The file “EDA_functions.py” is a Python script that contains all the functions used to clean the season data, combine the seasons into their respective game-state data frames and to build and access the players’ indices as well as the function to run the recommender engine itself.
## How to use the recommender:
The recommender uses the pairwise distances of a player’s index with reference to the game-state of a given season and returns the data for the 5 most similar players. The actual recommender is in the Jupyter Notebook called “All_Seasons.ipynb” which was also used to combine the game-states of each season into the data frames used in the recommender engine.
Steps to run the recommender:
### Step 1: Get the player’s index for the desired game-state and season.
To do this, run the function: get_index_all_gamestates(‘player’s full name you want recommended’). 
```python get_index_all_gamestates('Nick Suzuki') ```
The output will be a nested dictionary of the  looks liek this:
Nick Suzuki's ALL STRENGTHS indices are: {2022: [749], 2023: [1643], 2024: [2508]}
Nick Suzuki's EVEN STRENGTH indices are: {2022: [749], 2023: [1643], 2024: [2508]}
Nick Suzuki's POWER PLAY indices are: {2022: [695], 2023: [1510], 2024: [2316]}
Nick Suzuki's PENALTY KILL indices are: {2022: [681], 2023: [1467], 2024: [2224]}

### Step 2: *Optional*
Get the baseline stats for the player and game-state you want as the recommender's reference.
Run the function: get_players_baseline_gamestate_stats.
<img width="468" alt="image" src="https://github.com/ZachRosenthal04/Player_Recommender_NHL/assets/140099747/734dc59c-5ba6-44fa-af18-0d96a940e62e">
The output of this function is the original gamestate dataframe with all the player's stats for that gamestate.
The acceptable inputs for “original_gamestate_df” are: df_all_stats_AS, df_all_stats_ES, df_all_stats_PP, and df_all_stats_PK. 
If the player_name is misspelled or there is no data, the function returns an empty data frame.
### Step 3: Finding Similar NHL Players
Call the function: ```python recommend_skaters() ```
The function looks like this:
<img width="468" alt="image" src="https://github.com/ZachRosenthal04/Player_Recommender_NHL/assets/140099747/b5db79cf-9be4-41a5-8f26-be088dec73fc">
The sample output for Nick Suzuki's 2022 stats for even strength game states is:
<img width="468" alt="image" src="https://github.com/ZachRosenthal04/Player_Recommender_NHL/assets/140099747/f1844357-106b-44b3-a219-8af4f19c7220">
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


























