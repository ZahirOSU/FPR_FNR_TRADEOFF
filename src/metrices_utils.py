from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix


class MetricsUtil:

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
