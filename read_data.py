import pandas as pd

def load_df(data_type="train"):
    sales_attrs = [
        'Id',
    #     'SalePrice',
        'SaleCondition_Abnorml',
        'SaleCondition_AdjLand',
        'SaleCondition_Alloca',
        'SaleCondition_Family',
        'SaleCondition_Normal',
        'SaleCondition_Partial',
        'SaleType_COD',
        'SaleType_CWD',
        'SaleType_Con',
        'SaleType_ConLD',
        'SaleType_ConLI',
        'SaleType_ConLw',
        'SaleType_New',
        'SaleType_Oth',
        'SaleType_WD',
        ]
    
    df = pd.read_csv('cleaned_data.csv', parse_dates={'dt' : ['MoSold', 'YrSold']}, na_values=['NaN','?'], infer_datetime_format=True, low_memory=False, index_col='dt')
    for j in range(0,df.shape[1]):        
            df.iloc[:,j]=df.iloc[:,j].fillna(df.iloc[:,j].mean())
    df.drop(sales_attrs, axis=1, inplace=True)
    df_attrs = df.columns.tolist()
    df_attrs.remove('SalePrice')
    df = df[df_attrs + ['SalePrice']]
    df = df.sort_index()
    return df

def load_df():
    sales_attrs = [
        'Id',
    #     'SalePrice',
        'SaleCondition_Abnorml',
        'SaleCondition_AdjLand',
        'SaleCondition_Alloca',
        'SaleCondition_Family',
        'SaleCondition_Normal',
        'SaleCondition_Partial',
        'SaleType_COD',
        'SaleType_CWD',
        'SaleType_Con',
        'SaleType_ConLD',
        'SaleType_ConLI',
        'SaleType_ConLw',
        'SaleType_New',
        'SaleType_Oth',
        'SaleType_WD',
        ]
    
    df = pd.read_csv('cleaned_data.csv', parse_dates={'dt' : ['MoSold', 'YrSold']}, na_values=['NaN','?'], infer_datetime_format=True, low_memory=False, index_col='dt')
    for j in range(0,df.shape[1]):        
            df.iloc[:,j]=df.iloc[:,j].fillna(df.iloc[:,j].mean())
    df.drop(sales_attrs, axis=1, inplace=True)
    df_attrs = df.columns.tolist()
    df_attrs.remove('SalePrice')
    df = df[df_attrs + ['SalePrice']]
    df = df.sort_index()
    return df