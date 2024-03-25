import numpy as np
import pandas as pd
import copy
import re
from sklearn.model_selection import train_test_split, RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.conversions import StrToComposition


class VoltageMiningModel:
    def __init__(self, model) -> None:
        self.model = model
        self.final_model = None
        self.feature_list = None

    def insert_data(self, df, type="all") -> pd.DataFrame:
        # insert a df into the class's attribute
        if type == "all":
            self.data_all = df
            self.y_all = df.voltage
        if type == "train":
            self.data_train = df
            self.y_train = df.voltage
        if type == "test":
            self.data_test = df
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

    def get_feature_importance(self):
        model_type = copy.deepcopy(self.model)
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

    def get_top_features_list(self, n_features):
        feature_importance = self.get_feature_importance()
        self.feature_list = list(feature_importance["feature"][:n_features])

    def get_final_matrix(self, process_functions=None, type="all"):
        # take featurized df's and fill final df's for training after processing
        # processing functions take in entire df and return entire df
        if process_functions == None:
            process_functions = [lambda x: x]

        if type == "all":
            for function in process_functions:
                self.final_all = function(self.featurized_data_all)
                self.final_all = self.final_all[self.feature_list]
        if type == "train":
            for function in process_functions:
                self.final_train = function(self.featurized_data_train)
                self.final_train = self.final_train[self.feature_list]
        if type == "test":
            for function in process_functions:
                self.final_test = function(self.featurized_data_test)
                self.final_test = self.final_test[self.feature_list]

    def train_model(self):
        model = copy.deepcopy(self.model)
        model.fit(self.final_train, self.y_train)
        self.final_model = model
        return model

    def run_cv(self, n_repeats=5):
        # run cross validation on self.final_train
        model = copy.deepcopy(self.model)
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

    def pred_from_file(self, file_path, output_csv=False):
        pred_df = pd.read_csv(file_path, index_col=0)

        self.insert_data(pred_df, type="test")
        feature_list = [matminer_formula_feature("formula_charge"), matminer_formula_feature("formula_discharge")]
        self.featurize_df(feature_list, type="test") # get featurized test df

        self.get_final_matrix(type="test") # get final df for model prediction
        pred_result = self.final_model.predict(self.final_test) # do model prediction using the stored final model
        pred_df["predicted_voltage"] = pred_result

        if output_csv:
            output_path = re.sub(".csv", "_pred.csv", file_path)
            pred_df.to_csv(output_path)

        return pred_df

    @staticmethod
    def get_feature_matrix(orig_df, feature_functions):
        concat_df_list =[]
        for function in feature_functions:
            concat_df_list.append(function(orig_df))
        return pd.concat(concat_df_list, axis=1)

    @staticmethod
    def evaluate_model(model, X, y):
        r2, mae, rmse = [
            model.score(X, y),
            mean_absolute_error(y, model.predict(X)),
            root_mean_squared_error(y, model.predict(X)),
        ]
        return r2, mae, rmse


def matminer_formula_feature(formula):

    def matminer_formula(df, formula):
        df = df[[formula]].copy()
        df = StrToComposition().featurize_dataframe(df, formula, ignore_errors=True)

        ep_feat = ElementProperty.from_preset(preset_name="magpie")
        df = ep_feat.featurize_dataframe(df, col_id="composition")

        excluded = [formula, "composition"]
        feature_cols = df.drop(excluded, axis=1)

        prefix = ""
        if "charge" in formula:
            prefix = "chg_"
        if "discharge" in formula:
            prefix = "dis_"
        feature_cols = feature_cols.add_prefix(prefix)

        feature_cols = feature_cols.fillna(0)

        return feature_cols

    return lambda df: matminer_formula(df, formula)

