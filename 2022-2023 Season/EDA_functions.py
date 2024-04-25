#EDA_functions

import pandas as pd

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


def merge_similar_dfs(df1, df2, merge_col, how):
    """
    Merges two pandas DataFrames with similar columns by a specified column.
    
    This function identifies duplicate columns between two DataFrames and merges them on a specified column. 
    Duplicate columns, except for the merging column, are dropped from the second DataFrame to prevent 
    column duplication in the merged DataFrame. The resulting DataFrame will have its index reset to start at 1.

    Parameters:
    - df1 (pandas.DataFrame): The first DataFrame to merge.
    - df2 (pandas.DataFrame): The second DataFrame to merge. Duplicate columns, except for the merge column, will be dropped.
    - merge_col (str): The column name on which to merge the two DataFrames.
    - how (str): How to perform the merge. Can be one of 'left', 'right', 'outer', 'inner'.

    Returns:
    - pandas.DataFrame: The merged DataFrame containing combined data from df1 and df2 with index starting at 1.

    Raises:
    - ValueError: If 'merge_col' is not present in either DataFrame.
    - ValueError: If 'how' is not one of the expected merge strategies ('left', 'right', 'outer', 'inner').
    """
    
    valid_merge_methods = ['left', 'right', 'outer', 'inner']
    
    if how not in valid_merge_methods:
        raise ValueError(f"The 'how' parameter must be one of the following: {valid_merge_methods}")
    
    if merge_col not in df1.columns or merge_col not in df2.columns:
        raise ValueError(f"The specified 'merge_col' must be present in both DataFrames")

    try:
        duplicated_columns = df1.columns.intersection(df2.columns)
        duplicated_columns_to_drop = duplicated_columns.drop(merge_col)
        df2_with_duplicated_cols_dropped = df2.drop(columns=duplicated_columns_to_drop)
        df_merged = pd.merge(df1, df2_with_duplicated_cols_dropped, on=merge_col, how=how) 

        # Reset the index to start at 1 instead of the default 0.
        df_merged.reset_index(drop=True, inplace=True)
        df_merged.index += 1
        
    except KeyError as e:
        raise KeyError(f"Merge failed due to a KeyError: {e}. Ensure 'merge_col' is a column in both DataFrames.") from e
    
    return df_merged