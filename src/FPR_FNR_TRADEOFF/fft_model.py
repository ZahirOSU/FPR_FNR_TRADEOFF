from macest.classification.models import ModelWithConfidence
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import numpy as np
import copy


class FprFnrTradeoffModel(ModelWithConfidence):
    POSITIVE = 'Positive'
    NATURAL = 'Neutral'
    NEGATIVE = 'Negative'

    def __init__(self, *, threshold: float, **kwargs):
        assert 0 <= threshold <= 1, 'The threshold must be between 0 and 1'

        self.threshold = threshold

        super().__init__(**kwargs)

    @property
    def critic_incorrect_preds(self) -> str:
        # the threshold express the probability that the prediction is positive.
        if self.threshold > 0.5:
            # positive is more harmful > to remain positive a prediction need to pass high threshold.
            return FprFnrTradeoffModel.POSITIVE
        elif self.threshold < 0.5:
            # negative is more harmful > change to positive require passing low threshold.
            return FprFnrTradeoffModel.NEGATIVE
        else:  # self.threshold == 0.5
            return FprFnrTradeoffModel.NATURAL

    def predict_with_tradeoff(self, base_model_pred: np.ndarray, X_test: np.ndarray):
        preds = copy.deepcopy(base_model_pred)
        macest_conf_preds = self.predict_proba(X_test)

        if self.critic_incorrect_preds == FprFnrTradeoffModel.POSITIVE:
            # positive is more harmful > change only positive predictions to 0 if they are over the
            # threshold (the threshold is high as it is the probability to be 1)
            preds[(preds == 1) & (macest_conf_preds[:, 1] < self.threshold)] = 0
        elif self.critic_incorrect_preds == FprFnrTradeoffModel.NEGATIVE:
            # negative is more harmful > change only negative predictions to 1 if they are over the
            # threshold (the threshold is low as it is the probability to be 1)
            preds[(preds == 0) & (macest_conf_preds[:, 1] > self.threshold)] = 1
        return preds


    @staticmethod
    def _calc_fpr_fnr(y_true, y_pred):
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp/(fp+tn)
        fnr = fn/(tp+fn)
        return fpr, fnr

    @staticmethod
    def _calc_precision(y_true, y_pred):
        return precision_score(y_true, y_pred)

    @staticmethod
    def _calc_recall(y_true, y_pred):
        return recall_score(y_true, y_pred)

    @staticmethod
    def _calc_accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)
