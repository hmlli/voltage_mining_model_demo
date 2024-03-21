
from matminer.featurizers.composition.composite import ElementProperty
from matminer.featurizers.conversions import StrToComposition


def matminer_formula_feature(formula):

    def matminer_formula(df, formula):
        df = df[[formula]].copy()
        df = StrToComposition().featurize_dataframe(df, formula)

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