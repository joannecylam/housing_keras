import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data_processing_config import categr_fields, satisfaction, live_qua, eval_fields, sales_attrs


class PrepareData(object):
    '''
    Preprocess data and store it in dataframe
    '''
    def __init__(self):
        self.categr_fields = categr_fields
        self.eval_fields = eval_fields
        self.sales_attrs = sales_attrs
        self.model_attr = None

    def get_train_data(self, filepath="all/train.csv"):
        df = self.get_num_data(filepath=filepath, data_type="train")
        train_attrs = df.columns.tolist()
        train_attrs.remove("SalePrice")
        self.model_attr = train_attrs
        df.drop('Id', axis=1, inplace=True)
        return self.normalize_df(df), df[["SalePrice"]]

    def get_test_data(self, filepath="all/test.csv"):
        if not self.model_attr:
            raise ValueError("please load train data first")
        test_df = self.get_num_data(filepath=filepath, data_type="test")
        test_df_new = pd.DataFrame()
        for attr in self.model_attr:
            if attr in test_df.columns.tolist():
                test_df_new[attr] = test_df[attr]
            else:
                test_df_new[attr] = np.zeros(test_df.shape[0])
        item_id = test_df[['Id']]
        test_df_new.drop('Id', axis=1, inplace=True)
        return self.normalize_df(test_df_new), item_id

    def normalize_df(self, df):
        scaler = MinMaxScaler(feature_range=(0, 1))
        values = scaler.fit_transform(df.values)
        return pd.DataFrame(values, columns=df.columns.tolist())

    def get_num_data(self, filepath, data_type):
        df = pd.read_csv(filepath)
        bin_features = {}
        for field in self.categr_fields:
            attrs = df[field].unique()
            for attr in attrs:
                fieldname = field + "_" + str(attr)
                bin_vec = df[field] == attr
                bin_features[fieldname] = bin_vec.astype(int)
        bin_df = pd.DataFrame(bin_features)        
        # convert the other str fields
        df = self.convert_str_values(df)
        final_df = pd.concat([df, bin_df], axis=1)
        if data_type=="train":
            final_df = self.rearrage_order(final_df)

        for j in range(0, final_df.shape[1]):        
            final_df.iloc[:,j] = final_df.iloc[:,j].fillna(final_df.iloc[:,j].mean())
        return final_df

    def rearrage_order(self, df):
        list_attr = df.columns.tolist()
        list_attr.remove('SalePrice')
        return df[list_attr + ['SalePrice']]

    def convert_str_values(self, df):
        # drop the binary features
        df.drop(labels=self.categr_fields, axis=1, inplace=True)
        # convert other string fields
        for field, value_map in self.eval_fields.iteritems():
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

    def export_csv(df, filename, index=False):
        df.to_csv(filename, index=index)

if __name__ == "__main__":
    ppd = PrepareData()
    ppd.get_num_data()