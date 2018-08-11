
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
    "ALQ":  5,
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

sales_attrs = [
    'SalePrice',
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
    'SaleType_WD']