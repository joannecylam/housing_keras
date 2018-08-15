import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from data_processing_config import categr_fields, satisfaction, live_qua, eval_fields, sales_attrs


class PrepareData(object):
    '''
    Prepare training and testing data.
    '''
    def __init__(self):
        self.categr_fields = categr_fields
        self.eval_fields = eval_fields
        self.sales_attrs = sales_attrs
        self.model_attr = None

    def get_train_data(self, filepath="all/train.csv", drop_cols=['Id'], time_series=False):
        df = self.get_num_data(filepath=filepath, data_type="train", time_series=time_series)
        df.drop(drop_cols, axis=1, inplace=True)
        train_attrs = df.columns.tolist()
        # store attributes of training data as obj property
        if 'SalePrice' in train_attrs:
            train_attrs.remove('SalePrice')
        self.model_attr = train_attrs

        return self.normalize_df(df), df[['SalePrice']]

    def get_test_data(self, filepath="all/test.csv", time_series=False):
        if not self.model_attr:
            raise ValueError("please load train data first")
        test_df = self.get_num_data(filepath=filepath, data_type="test", time_series=time_series)
        test_df_new = pd.DataFrame()
        for attr in self.model_attr:
            if attr in test_df.columns.tolist():
                test_df_new[attr] = test_df[attr]
            else:
                test_df_new[attr] = np.zeros(test_df.shape[0])
        item_id = test_df[['Id']]
        # test_df_new.drop('Id', axis=1, inplace=True)
        return self.normalize_df(test_df_new), item_id

    def normalize_df(self, df):
        scaler = MinMaxScaler(feature_range=(0, 1))
        values = scaler.fit_transform(df.values)
        return pd.DataFrame(values, columns=df.columns.tolist(), index=df.index)

    def get_num_data(self, filepath, data_type, time_series):
        if time_series:
            df = pd.read_csv(
                filepath, parse_dates={'dt': ['MoSold', 'YrSold']},
                na_values=['NaN', '?'], infer_datetime_format=True,
                low_memory=False, index_col='dt')
            df.sort_index(inplace=True)
        else:
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
        if data_type == "train":
            final_df = self.rearrage_order(final_df)

        for j in range(0, final_df.shape[1]):
            final_df.iloc[:, j] = final_df.iloc[:, j].fillna(final_df.iloc[:, j].mean())
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
            value_arr = [aver_v if x == "NA" else x for x in tmp]
            df[field] = value_arr
        return df

    def export_csv(self, df, filename, index=False):
        df.to_csv(filename, index=index)

if __name__ == "__main__":
    ppd = PrepareData()
    ppd.get_num_data()
