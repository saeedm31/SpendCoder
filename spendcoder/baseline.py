"""The baseline to beat: TF-IDF features and logistic regression.

Always start here. It trains in seconds on a laptop, is easy to explain, and is often strong on short texts.
Fine-tuning a transformer is only worth its cost if it clearly beats this.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, make_pipeline

from .taxonomy import CATEGORIES


class TfidfBaseline:
    name = "tfidf_logreg"

    def __init__(self, C=4.0):
        self.pipe = make_pipeline(
            FeatureUnion([
                ("words", TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True)),
                ("chars", TfidfVectorizer(lowercase=True, analyzer="char_wb", ngram_range=(3, 5), min_df=2,
                                          sublinear_tf=True)),
            ]),
            LogisticRegression(C=C, max_iter=2000, class_weight="balanced"),
        )

    def fit(self, texts, labels):
        self.pipe.fit(list(texts), np.asarray(labels))
        return self

    def predict_proba(self, texts):
        """Probabilities for all 14 categories, even ones missing from training (they get 0)."""
        p = self.pipe.predict_proba(list(texts))
        out = np.zeros((len(p), len(CATEGORIES)))
        out[:, self.pipe.classes_] = p
        return out
