import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold

import math
from typing import Dict


class Normalization:
    @staticmethod
    def min_max_scaler(feat_col: pd.Series) -> float:
        min_val = feat_col.min()
        max_val = feat_col.max()

        numerator_term = feat_col - min_val
        denominator_term = max_val - min_val

        return numerator_term / denominator_term

    @staticmethod
    def standardize(feat_col: pd.Series) -> float:
        mean = feat_col.mean()
        std = feat_col.std()

        return (feat_col - mean) / std


class DimReduction:
    def __init__(self):
        self.sorted_eiglist_desc: list[tuple[float, np.ndarray]] = None
        self.projection_matrix: np.ndarray = None

    def make_structs(self):
        pass

    def eig_decomposition(
        self, mtrx: np.ndarray, scree_plot: bool = True
    ) -> list[tuple[float, np.ndarray]]:
        # if np.linalg.det(mtrx) == 0:
        #     raise ValueError("Input matrix is singular and cannot be inverted.")

        eigenvalues, eigenvectors = np.linalg.eig(mtrx)

        eig_list = [
            (eigval, eigvect)
            for eigval, eigvect in zip(eigenvalues.tolist(), eigenvectors.tolist())
        ]
        self.sorted_eiglist_desc = sorted(eig_list, key=lambda x: x[0], reverse=True)

        if scree_plot:
            sum_eigenvalues = eigenvalues.sum()
            var_percentage = [
                (e[0] / sum_eigenvalues) * 100 for e in self.sorted_eiglist_desc
            ]

            pc_num = np.arange(1, len(eigenvalues) + 1)
            plt.bar(pc_num, var_percentage)
            plt.plot(pc_num, var_percentage, marker="o", linestyle="-", color="red")
            plt.xlabel("Number of Components")
            plt.ylabel("percentage of variance")
            plt.title("Scree Plot")
            plt.show()

        return self.sorted_eiglist_desc

    def project(self, datapoints: np.ndarray, k: int):
        # assort eigenvectors as columns of (master) projection matrix
        self.projection_matrix = np.array([e[1] for e in self.sorted_eiglist_desc])

        # apply the transformation to the feature columns, keeping the label column unchanged
        projected_arr = np.dot(datapoints, self.projection_matrix[:, :k])
        projected_df_dict = {f"PC{i+1}": projected_arr[:, i] for i in range(k)}
        projected_df_dict.update({"label": df["label"]})
        projected_df = pd.DataFrame(projected_df_dict)

        return projected_df


class PCA(DimReduction):
    def make_structs(self, df: pd.DataFrame) -> pd.DataFrame:
        return df[df.columns[:4]].cov().values


class FLD(DimReduction):
    def make_structs(self, df: pd.DataFrame):
        sw = np.zeros((4, 4))
        sb = np.zeros((4, 4))

        # rows: as many as distinct classes, columns: as many as features
        class_means = df.groupby("label").mean()
        # overall_mean = df[df.columns[:n_feat]].mean()
        overall_mean = df.iloc[:, :-1].mean()

        for c_label, c_mean in class_means.iterrows():
            diff = c_mean - overall_mean
            sb += np.outer(diff, diff.T) * len(df[df["label"] == c_label])

        for c in df["label"].unique():
            class_df = df[df["label"] == c]
            # class_cov = class_df[class_df.columns[:4]].cov().values
            class_cov = np.cov(class_df[class_df.columns[:4]].values.T)
            sw += class_cov

        m = np.linalg.inv(sw).dot(sb)

        return m, sw, sb, class_means, overall_mean


class NBGaussian:
    def __init__(self, inv_cov: np.ndarray, det_cov: int, class_mean: np.ndarray):
        self.inv_cov = inv_cov
        self.det_cov = det_cov
        self.class_mean = class_mean
        self.term_1 = 0.5 * math.log(det_cov, math.e)

    def g(self, sample: np.ndarray) -> float:
        diff = sample - self.class_mean
        term_2 = 0.5 * np.dot(diff.T, np.dot(self.inv_cov, diff))

        return -(self.term_1 + term_2)


class BayesClassifier:
    def __init__(self):
        self.class_discriminator_dict = {}

    def fit(self, train_df: pd.Series, k: int) -> Dict[str, NBGaussian]:
        for c in train_df["label"].unique():
            class_df = train_df[train_df["label"] == c]
            # class_cov = np.cov(class_df[class_df.columns[:k]].values.T)
            class_cov = class_df[class_df.columns[:k]].cov().values
            class_mean = class_df[class_df.columns[:k]].values.mean()

            class_discriminator = NBGaussian(
                inv_cov=np.linalg.inv(class_cov),
                det_cov=np.linalg.det(class_cov),
                class_mean=class_mean,
            )

            self.class_discriminator_dict[c] = class_discriminator

        return self.class_discriminator_dict

    def _row_predictor(self, sample: pd.Series) -> str:
        chances = {}
        for label, label_f in self.class_discriminator_dict.items():
            g = label_f.g(sample.to_numpy())
            chances[g] = label
        return chances.get(max(chances.keys()))

    def predict(self, test_df: pd.DataFrame, k: int) -> pd.DataFrame:
        if not self.class_discriminator_dict:
            raise ValueError("model is not fitted yet.")

        prediction_df = test_df.iloc[:, :k].apply(
            lambda row: self._row_predictor(row), axis=1
        )
        prediction_df.rename("Result")  # Rename the column
        return prediction_df


class Helpers:
    @staticmethod
    def df_splitter(df: pd.DataFrame, test_frac: float):
        if test_frac >= 1:
            raise ValueError("Test Ratio must be less than zero")

        test_df = df.sample(frac=test_frac)
        train_df = df.drop(test_df.index)

        return train_df, test_df

    @staticmethod
    def accuracy_score(target_labels: np.ndarray, predicted_labels: np.ndarray):
        correct_predictions = np.sum(target_labels == predicted_labels)
        total_predictions = len(target_labels)
        accuracy = correct_predictions / total_predictions

        return accuracy


def run(df_normalized: pd.DataFrame, k: int, instance: FLD | PCA):
    m = instance.make_structs(
        df=df_normalized,
    )
    if isinstance(m, tuple):
        m = m[0]

    instance.eig_decomposition(mtrx=m, scree_plot=False)
    projected_df = instance.project(datapoints=df_normalized.values[:, :4], k=k)

    kf = KFold(n_splits=4, shuffle=True, random_state=42)

    accuracies = []
    for train_index, test_index in kf.split(projected_df):
        train_df, test_df = (
            projected_df.iloc[train_index],
            projected_df.iloc[test_index],
        )

        clf = BayesClassifier()
        clf.fit(train_df=train_df, k=k)

        pred_df = clf.predict(test_df=test_df, k=k)

        # Calculate accuracy for this fold
        accuracy = Helpers.accuracy_score(
            target_labels=test_df.values[:, k], predicted_labels=pred_df.values
        )
        accuracies.append(accuracy)

    return accuracies


if __name__ == "__main__":
    feat_names = ["sepal_length", "sepal_width", "petal_length", "petal_width", "label"]
    df = pd.read_csv(
        "mini_proj2/iris.csv",
        sep=",",
        names=feat_names,
    )
    df_normalized = df
    # df_normalized[df.columns[:4]] = df[df.columns[:4]].apply(min_max_scaler)
    df_normalized[df.columns[:4]] = df[df.columns[:4]].apply(Normalization.standardize)

    pca = PCA()
    fld = FLD()

    for k in range(1, df.shape[1] - 1):
        print(f"#principal component selected = {k}")

        pca_accuracies = run(df_normalized=df_normalized, k=k, instance=pca)
        fld_accuracies = run(df_normalized=df_normalized, k=k, instance=fld)

        print(
            f"PCA avg accuracy  per {len(pca_accuracies)} runs= {sum(pca_accuracies)/len(pca_accuracies)}"
        )
        print(
            f"FLD avg accuracy  per {len(fld_accuracies)} runs= {sum(fld_accuracies)/len(fld_accuracies)}"
        )
