import numpy as np
import pandas as pd

categr_fields = [
    "MSSubClass",
    "MSZoning",
    "Street",
    "Alley",
    "LotShape",
    "LandContour",
    "Utilities",
    "LotConfig",
    "LandSlope",
    "Neighborhood",
    "Condition1",
    "Condition2",
    "BldgType",
    "HouseStyle",
    "RoofStyle",
    "RoofMatl",
    "Exterior1st",
    "Exterior2nd",
    "MasVnrType",
    "Foundation",
    "Heating",
    "CentralAir",
    "Electrical",
    "GarageType",
    "SaleType",
    "MiscFeature",
    "SaleCondition"
]


# In[21]:


satisfaction = {
        "Ex": 5,
        "Gd": 4,
        "TA": 3,
        "Fa": 2,
        "Po": 1,
        "NA": 0,
    }


live_qua = {
    "GLQ":6,
    "ALQ":	5,
    "BLQ": 4,
    "Rec" : 3,
    "LwQ": 2,
    "Unf": 1,
    "NA": 0
}

eval_fields = {
    "ExterQual":satisfaction,
    "ExterCond": satisfaction,
    "BsmtQual": satisfaction,
    "BsmtCond": satisfaction,
    "BsmtExposure": {
        "Gd": 4,
        "Av": 3,
        "Mn": 2,
        "No": 1,
        "NA": 0,
    } ,
    "BsmtFinType1": live_qua,
    "BsmtFinType2": live_qua,
    "HeatingQC": satisfaction,
    "KitchenQual": satisfaction,
    "Functional":{
        "Typ": 7,
        "Min1": 6,
        "Min2": 5,
        "Mod": 4,
        "Maj1": 3,
        "Maj2": 2,
        "Sev": 1,
        "Sal": 0
    },
    "FireplaceQu": satisfaction,
    "GarageFinish":{
        "Fin":3,
        "RFn": 2,
        "Unf": 1,
        "NA": 0
    },
    "GarageQual":satisfaction,
    "GarageCond":satisfaction,
    "PavedDrive":{
        "Y": 2,
        "P": 1,
        "N": 0
    },
    "PoolQC":satisfaction,
    "Fence":{
        "GdPrv": 2,
        "MnPrv": 1,
        "GdWo": 2,
        "MnWw": 1,
        "NA": 0
    },
}

def get_bin_values(df, categr_fields):
    bin_features = {}
    for field in categr_fields:
        attrs = df[field].unique()
        for attr in attrs:
            fieldname = field + "_" + str(attr)
            bin_vec = df[field] == attr
            bin_features[fieldname] = bin_vec.astype(int)
    return bin_features


def convert_str_values(df, eval_fields):
    for field, value_map in eval_fields.iteritems():
        tmp = []
        non_zeros = []
        for v in df[field]:
            if value_map.get(v):
                value = value_map[v]
                tmp.append(value)
                non_zeros.append(value)
            else:
                tmp.append("NA")
        aver_v = np.average(non_zeros)
        # assign average value to None values
        value_arr = [aver_v if x=="NA" else x for x in tmp]
        df[field] = value_arr
    return df

if __name__ == "__main__":

    df = pd.read_csv("all/train.csv")
    bin_features = get_bin_values(df, categr_fields)
    bin_df = pd.DataFrame.from_dict(bin_features)
    # drop the binary features
    df = df.drop(labels=categr_fields, axis=1)
    # convert the other str fields
    convert_str_values(df, eval_fields)
    # combine with binary feature vectors
    final_df = pd.concat([df, bin_df], axis=1)
    final_df.to_csv("cleaned_data.csv")

