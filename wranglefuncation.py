import pandas as pd
def wrangle(filepath: str):
    """Wrangle funcation to clean and preprocess a dataset.
    
    Args:
    ----
    filepath (str): The filepath of the dataframe to be cleaned
    
    Return:
    ------
    A Cleaned dataframe based on the insights from analysis.
    """
    # Load the data and check for the path if it correct 
    df = pd.read_csv(filepath) 
    # 1. Drop the unused columns 
    df.drop(columns=["customer_id", "date_of_registration", "pincode"], inplace=True)
    
    # 3. Processing the false values (Negative) values in the data_used column
    mean_value = round(df["data_used"].mean())
    df["data_used"] = df["data_used"].apply(lambda x: mean_value if x < 0 else x)

    
    return df    
