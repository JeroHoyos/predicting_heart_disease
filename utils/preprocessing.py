from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd

class OutlierRemoverIQR(BaseEstimator, TransformerMixin):

    def fit(self, df, numerical_columns):
        if isinstance(numerical_columns, str):
            self.numerical_columns_ = [numerical_columns]
        else:
            self.numerical_columns_ = numerical_columns

        self.stats_ = pd.DataFrame(index=self.numerical_columns_)
        self.stats_["Q1"] = df[self.numerical_columns_].quantile(0.25)
        self.stats_["Q3"] = df[self.numerical_columns_].quantile(0.75)
        self.stats_["IQR"] = self.stats_["Q3"] - self.stats_["Q1"]
        self.stats_["lower_cutoff"] = self.stats_["Q1"] - 1.5 * self.stats_["IQR"]
        self.stats_["upper_cutoff"] = self.stats_["Q3"] + 1.5 * self.stats_["IQR"]

        return self

    def transform(self, df):
        masks = (
            (df[self.numerical_columns_] >= self.stats_["lower_cutoff"]) &
            (df[self.numerical_columns_] <= self.stats_["upper_cutoff"])
        )
        final_mask = masks.all(axis=1)
        return df[final_mask]
