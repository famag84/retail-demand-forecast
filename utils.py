from typing import List
import pandas as pd
from process import * 


def create_target(data:pd.DataFrame, lag:int = 28) -> pd.DataFrame:
    """ 
    Creates Target as a new columns with name "demand". 
    Returns input data with new column
    """
    data = data.sort_values(by=["item_id", "store_id", "date"])
    data[f"demand"] = data.groupby(["item_id", "store_id"], observed=False)["sales"].shift(-lag).fillna(0)
    return data



def preprocess_wm_yr_wk(df):
    df["year"] = df["wm_yr_wk"].apply(lambda x: int(str(x)[1:3]))
    df["wknu"] = df["wm_yr_wk"].apply(lambda x: int(str(x)[-2:]))
    df["pack4wk"] = df["wknu"]//4
    return df


def get_custom_calendar():
    snap_cols = ["snap_CA", "snap_TX", "snap_WI"]
    cat_cols = ['event_name_1', 'event_type_1'] #, 'event_name_2', 'event_type_2']
    force_column_order = ["d", "date", "wm_yr_wk", "year", "wknu"] + cat_cols
    
    calendar = load_calendar(PATH_INPUT)
    calendar = preprocess_wm_yr_wk(calendar)
    return calendar[force_column_order]


def get_input_data(data:pd.DataFrame, prices: pd.DataFrame, calendar: pd.DataFrame, start_date: str | None = None, state_id: str =  "CA", item_id: str | None = None, drop_columns: List | None = None) -> pd.DataFrame:
    """
    calendar: pandas dataframe with signicant events.
    start_date: date format "yyy-mm-dd"
    drop_columns: list of column names to drop 
    """
    drop_columns = drop_columns if drop_columns else []
    data = data.set_index(id_cols).stack().reset_index().rename(columns={"level_6": "d", 0: "sales"})
    
    # We kept data only for state CA tryind to reduce the volume of data in memory.
    cond = (data["state_id"]==state_id)
    if item_id: 
        cond = cond & (data["item_id"]==item_id)
        
    data = data[cond]
    initial_shape = len(data)
    calendar_cols = ['d', 'date', "wm_yr_wk", 'wknu', 'event_type_1', "event_name_1"]
    data = pd.merge(left=data, right=calendar[calendar_cols], left_on="d", right_on="d", how="left")
    data = pd.merge(left=data, right=prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    
    # Keep data since start_date
    if start_date:
        data = data[data["date"]>=start_date]

    selected_columns = id_cols + calendar_cols + ["sales"] 
    selected_columns = [c for c in selected_columns if c not in drop_columns]
    
    # Ensure sorted data
    data = data.sort_values(["date", "store_id", "item_id"], ascending=True)
    
    return data[selected_columns].copy()

