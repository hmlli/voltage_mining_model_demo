import numpy as np
import pandas as pd
import copy
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error


class VoltageMiningModel:
    def __init__(self) -> None:
        pass

    def insert_data(self, df, type="all") -> pd.DataFrame:
        # insert a df into the class's attribute
        if type == "all":
            self.data_all = df
            self.y_all = df.voltage
        if type == "train":
            self.data_train=df
            self.y_train = df.voltage
        if type == "test":
            self.data_test=df
            if "voltage" in df.keys():
                self.y_test = df.voltage

    def featurize_df(self, feature_list, type="all"):
        # use a list of feature functions to featurize one of the df's stored
        if type == "all":
            self.featurized_data_all = self.get_feature_matrix(self.data_all, feature_list)
        if type == "train":
            self.featurized_data_train = self.get_feature_matrix(self.data_train, feature_list)
        if type == "test":
            self.featurized_data_test = self.get_feature_matrix(self.data_test, feature_list)

    def train_test_split(self, test_size=0.2, stratify=None):
        # do train/test splitting and fill featurized_train and _test
        # only works when self.data_all is not None

        if test_size == 0:
            self.featurized_data_train = self.featurized_data_all
            self.y_train = self.y_all
            self.featurized_data_test = None
            self.y_test = None

        else:
            if not stratify:
                self.featurized_data_train, self.featurized_data_test, self.y_train, self.y_test = train_test_split(self.featurized_data_all, self.y_all, test_size=test_size)
            if stratify == "element":
                df_train, df_test, _, _ = train_test_split(self.data_all, self.data_all["element"], test_size=test_size, stratify=self.data_all["element"])
                index_train, index_test = df_train.index, df_test.index
                self.featurized_data_train, self.y_train = self.featurized_data_all.loc[index_train], self.y_all[index_train]
                self.featurized_data_test, self.y_test = self.featurized_data_all.loc[index_test], self.y_all[index_test]

    def get_feature_importance(self, model_type):
        model_type = copy.deepcopy(model_type)
        model = Pipeline(
            [
                ("imputer", SimpleImputer()),  # For the failed structures
                ("model", model_type)
            ]
        )
        model.fit(self.featurized_data_train, self.y_train)
        feature_importances_rf = pd.DataFrame(
            {
                "feature": model[0].get_feature_names_out(),
                "importance": model[-1].feature_importances_,
            }
        ).sort_values(by="importance", ascending=False)
        return feature_importances_rf

    def get_final_matrix(self, process_functions=None, type="all"):
        # take featurized df's and fill final df's for training after processing
        # processing functions take in entire df and return entire df
        if process_functions == None:
            process_functions = [lambda x: x]

        if type == "all":
            for function in process_functions:
                self.final_all = function(self.featurized_data_all)
        if type == "train":
            for function in process_functions:
                self.final_train = function(self.featurized_data_train)
        if type == "test":
            for function in process_functions:
                self.final_test = function(self.featurized_data_test)

    def train_model(self, model):
        model = copy.deepcopy(model)
        model.fit(self.final_train, self.y_train)
        self.final_model = model
        return model

    def run_cv(self, model, n_repeats=5):
        # run cross validation on self.final_train
        model = copy.deepcopy(model)
        rpf = RepeatedKFold(n_repeats=n_repeats)
        train_r2s, train_maes, train_rmses, test_r2s, test_maes, test_rmses = [], [], [], [], [], []

        for train, test in rpf.split(self.final_train):
            X_train, X_test = self.final_train.iloc[train], self.final_train.iloc[test]
            y_train, y_test = self.y_train.iloc[train], self.y_train.iloc[test]

            model.fit(X_train, y_train)
            train_r2, train_mae, train_rmse = self.evaluate_model(model, X_train, y_train)
            test_r2, test_mae, test_rmse, = self.evaluate_model(model, X_test, y_test)

            train_r2s.append(train_r2)
            train_maes.append(train_mae)
            train_rmses.append(train_rmse)
            test_r2s.append(test_r2)
            test_maes.append(test_mae)
            test_rmses.append(test_rmse)

        return np.average(train_r2s), np.average(train_maes), np.average(train_rmses), np.average(test_r2s), np.average(test_maes), np.average(test_rmses)

    @staticmethod
    def get_feature_matrix(orig_df, feature_functions):
        concat_df_list =[]
        for function in feature_functions:
            concat_df_list.append(function(orig_df))
        return pd.concat(concat_df_list, axis=1)

    @staticmethod
    def select_top_features(feature_df, n):
        return lambda X: X[feature_df["feature"][:n]]

    @staticmethod
    def evaluate_model(model, X, y):
        r2, mae, rmse = [
            model.score(X, y),
            mean_absolute_error(y, model.predict(X)),
            mean_squared_error(y, model.predict(X), squared=False),
        ]
        return r2, mae, rmse

