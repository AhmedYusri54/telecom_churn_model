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
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print("Invalid filepath try a vaild one...")
    else:
        # print("DataLoaded and will been process")
        pass
        
    # 1. Drop the unused columns 
    df.drop(columns=["customer_id", "date_of_registration", "pincode"], inplace=True)
    
    # 2. Change the targets columns from object type into int type 
    df["gender"] = df["gender"].map({"F": 0, "M": 1})
    
    # 3. Processing the false values (Negative) values in the data_used column
    mean_value = round(df["data_used"].mean())
    df["data_used"] = df["data_used"].apply(lambda x: mean_value if x < 0 else x)

    
    return df    
