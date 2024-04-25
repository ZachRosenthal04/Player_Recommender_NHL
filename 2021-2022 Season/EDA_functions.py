#EDA_functions

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