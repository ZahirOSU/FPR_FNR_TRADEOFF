from macest.classification import models as macest_model
from sklearn.metrics import precision_score, recall_score, accuracy_score


class FprFnrTradeoffModel:

    def __init__(self, model, X_conf_train, y_conf_train, threshold: float, critical: str = 'positive'):
        assert 0 <= threshold < 1, 'the threshold must be between 0 and 1'
        assert critical in ['positive', 'negative'], "dont_miss must be \'positive\' or \'negative\'"
        self.model = model
        self.macest_model = macest_model.ModelWithConfidence(model,
                                                 X_conf_train,
                                                 y_conf_train)
        self.threshold = threshold
        self.critical = critical

    def fit(self, X_cal, y_cal):
        self.macest_model.fit(X_cal, y_cal)

    def predict(self, X_test):
        xgboost_preds = self.model.predict(X_test)
        macest_conf_preds = self.macest_model.predict_proba(X_test)
        # xgboost_conf_preds = model.predict_proba(X_test)
#         todo: if positive: take all class 0 predictions that are above the 1-threshold -> change to 1
#           if negative: take all class 1 predictions that are under the threshold > change to 0
        if self.critical == 'positive':
            xgboost_preds[(xgboost_preds == 0) & (macest_conf_preds[:, 1] > 1-self.threshold)] = 1
        if self.critical == 'negative':
            xgboost_preds[(xgboost_preds == 1) & (macest_conf_preds[:, 1] < self.threshold)] = 0

    @staticmethod
    def _calc_false_positive_rate(y_true, y_pred):
        pass

    @staticmethod
    def _calc_false_negative_rate(y_true, y_pred):
        pass

    @staticmethod
    def _calc_precision(y_true, y_pred):
        return precision_score(y_true, y_pred)

    @staticmethod
    def _calc_recall(y_true, y_pred):
        return recall_score(y_true, y_pred)

    @staticmethod
    def _calc_accuracy(y_true, y_pred):
        return accuracy_score(y_true, y_pred)

    @staticmethod
    def calculate_metrics(y_true, y_pred):
        false_positive_rate = FprFnrTradeoffModel._calc_false_positive_rate(y_true, y_pred)
        false_negative_rate = FprFnrTradeoffModel._calc_false_negative_rate(y_true, y_pred)
        precision = FprFnrTradeoffModel._calc_precision(y_true, y_pred)
        recall = FprFnrTradeoffModel._calc_recall(y_true, y_pred)



        
