from metrics.metric_abstract import MetricAbstract, MetricSingleSlice
from metrics.metric_abstract import MetricCI_clopper_pearson, MetricCI_bootstrap, MetricCI_given, MetricCI_normal, MetricCI_none
from metrics.metric_abstract import MetricChained
from data_slicing.data_slice import DataSlice


class AUC(MetricSingleSlice, MetricCI_normal):
    """Area Under the ROC Curve"""
    from sklearn.metrics import roc_auc_score
    name = 'AUC'

    @classmethod
    def metric_function(cls, s: DataSlice, Y: tuple, *args, **kwargs) -> DataSlice:
        # from settings.settings_data_slicing_pipeline import settings
        import numpy as np
        classes = cls.settings.get('evaluation').get('binary_classification').get('classes')
        if len(Y[0].unique()) == 1 or len(Y[1].unique()) == 1:
            auc = np.nan
        else:
            auc = cls.roc_auc_score(*Y, *args, labels=classes)
        s.update_metric_value(cls.name, auc)
        return s

    @classmethod
    def calc_CI(cls, s: DataSlice, Y: tuple, *args, **kwargs) -> DataSlice:
        from roc_comparison.compare_auc_delong_xu import delong_roc_variance  # from https://github.com/yandexdataschool/roc_comparison.git
        import numpy as np
        if len(Y[0].unique()) == 1 or len(Y[1].unique()) == 1:
            return MetricCI_none.calc_CI(s, name=cls.name)
        auc, auc_cov = delong_roc_variance(*Y)
        auc_std = np.sqrt(auc_cov)
        return super().calc_CI(s, auc, auc_std)


class AveragePrecision(MetricSingleSlice, MetricCI_bootstrap):
    """Average Precision"""
    from sklearn.metrics import average_precision_score
    name = 'AP'

    @classmethod
    def metric_function(cls, s: DataSlice, Y: tuple, *args, **kwargs) -> DataSlice:
        s.update_metric_value(cls.name, cls.average_precision_score(*Y))
        return s


class NPV(MetricSingleSlice, MetricCI_clopper_pearson, MetricChained):
    """Negative Predictive Value tn / tn + fn"""
    name = 'NPV'

    @classmethod
    def metric_function(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        s = cls.chain(s, ConfusionMatrix, *args, **kwargs)
        s.update_metric_value(cls.name, s.get_metric_value('tn') / s.get_metric_value('pn'))
        return s

    @classmethod
    def calc_CI(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        return super().calc_CI(s, 'tn', 'pn', *args, **kwargs)


class Specificity(MetricSingleSlice, MetricCI_clopper_pearson, MetricChained):
    name = 'specificity'

    @classmethod
    def metric_function(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        s = cls.chain(s, ConfusionMatrix, *args, **kwargs)
        s.update_metric_value(cls.name, s.get_metric_value('tn') / s.get_metric_value('n'))
        return s

    @classmethod
    def calc_CI(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        return super().calc_CI(s, 'tn', 'n', *args, **kwargs)


class TNR(Specificity):
    """True Negative Rate = Specificity"""
    name = 'TNR'


class Selectivity(Specificity):
    """Selectivity = Specificity"""
    name = 'selectivity'


class Phi(MetricSingleSlice, MetricCI_bootstrap, MetricChained):
    """Phi or Matthews Correlation Coefficient"""
    name = 'phi'

    @classmethod
    def metric_function(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        import numpy as np
        s = cls.chain(s, ConfusionMatrix, *args, **kwargs)
        tp = s.get_metric_value('tp')
        fp = s.get_metric_value('fp')
        tn = s.get_metric_value('tn')
        fn = s.get_metric_value('fn')
        if (tp + fp) == 0 or (tp + fn) == 0 or (tn + fp) == 0 or (tn + fn) == 0:
            phi = 0
        else:
            phi = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        s.update_metric_value(cls.name, phi)
        return s


class Precision(MetricSingleSlice, MetricCI_clopper_pearson, MetricChained):
    name = 'precision'

    @classmethod
    def metric_function(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        s = cls.chain(s, ConfusionMatrix, *args, **kwargs)
        s.update_metric_value(cls.name, s.get_metric_value('tp') / s.get_metric_value('pp'))
        return s

    @classmethod
    def calc_CI(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        return super().calc_CI(s, 'tp', 'pp', *args, **kwargs)


class PPV(Precision):
    """Positive Predictive Value = Precision"""
    name = 'ppv'


class Recall(MetricSingleSlice, MetricCI_clopper_pearson, MetricChained):
    name = 'recall'

    @classmethod
    def metric_function(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        s = cls.chain(s, ConfusionMatrix, *args, **kwargs)
        s.update_metric_value(cls.name, s.get_metric_value('tp') / s.get_metric_value('p'))
        return s

    @classmethod
    def calc_CI(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        return super().calc_CI(s, 'tp', 'p', *args, **kwargs)


class Sensitivity(Recall):
    """Sensitivity = Recall"""
    name = 'sensitivity'


class HitRate(Recall):
    """Hit Rate = Recall"""
    name = 'hit_rate'


class TPR(Recall):
    """True Positive Rate = Recall"""
    name = 'TPR'


class F1(MetricSingleSlice, MetricCI_bootstrap, MetricChained):
    name = 'F1'

    @classmethod
    def metric_function(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        s = cls.chain(s, Precision, *args, **kwargs)
        s = cls.chain(s, Recall, *args, **kwargs)
        s.update_metric_value(cls.name, 2 * (s.get_metric_value(Recall.name) * s.get_metric_value(Precision.name)) / \
            (s.get_metric_value(Precision.name) + s.get_metric_value(Recall.name)))
        return s


class F1Complement(MetricSingleSlice, MetricCI_bootstrap, MetricChained):
    name = 'F1_complement'

    @classmethod
    def metric_function(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        s = cls.chain(s, NPV, *args, **kwargs)
        s = cls.chain(s, Specificity, *args, **kwargs)
        s.update_metric_value(cls.name, 2 * (s.get_metric_value(Specificity.name) * s.get_metric_value(NPV.name)) / \
            (s.get_metric_value(NPV.name) + s.get_metric_value(Specificity.name)))
        return s


class FNR(MetricSingleSlice, MetricCI_clopper_pearson, MetricChained):
    """False Negative Rate fn / p"""
    name = 'FNR'

    @classmethod
    def metric_function(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        s = cls.chain(s, ConfusionMatrix, *args, **kwargs)
        s.update_metric_value(cls.name, s.get_metric_value('fn') / s.get_metric_value('p'))
        return s

    @classmethod
    def calc_CI(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        return super().calc_CI(s, 'fn', 'p', *args, **kwargs)


class MissRate(FNR):
    """Miss Rate = False Negative Rate"""
    name = 'miss_rate'


class FPR(MetricSingleSlice, MetricCI_clopper_pearson, MetricChained):
    """False Positive Rate fp / n"""
    name = 'FPR'

    @classmethod
    def metric_function(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        s = cls.chain(s, ConfusionMatrix, *args, **kwargs)
        s.update_metric_value(cls.name, s.get_metric_value('fp') / s.get_metric_value('n'))
        return s

    @classmethod
    def calc_CI(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        return super().calc_CI(s, 'fp', 'n', *args, **kwargs)


class Informedness(MetricSingleSlice, MetricCI_bootstrap, MetricChained):
    """Informedness TPR - FPR"""
    name = 'informedness'

    @classmethod
    def metric_function(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        s = cls.chain(s, TPR, *args, **kwargs)
        s = cls.chain(s, FPR, *args, **kwargs)
        s.update_metric_value(cls.name, s.get_metric_value(TPR.name) / s.get_metric_value(FPR.name))
        return s


class Markedness(MetricSingleSlice, MetricCI_bootstrap, MetricChained):
    """Markedness tp / pp - tn / pn"""
    name = 'markedness'

    @classmethod
    def metric_function(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        s = cls.chain(s, ConfusionMatrix, *args, **kwargs)
        s.update_metric_value(cls.name, s.get_metric_value('tp') / s.get_metric_value('pp') - s.get_metric_value('tn') / s.get_metric_value('pn'))
        return s


class BalancedInformednessMarkedness(MetricSingleSlice, MetricCI_bootstrap, MetricChained):
    """Balanced markedness and informedness"""
    name = 'balanced_informedness_markedness'

    @classmethod
    def metric_function(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        s = cls.chain(s, Informedness, *args, **kwargs)
        s = cls.chain(s, Markedness, *args, **kwargs)
        s.update_metric_value(cls.name, 0.5 * (s.get_metric_value(Informedness.name) + s.get_metric_value(Markedness.name)))
        return s


class JaccardComplement(MetricSingleSlice, MetricCI_bootstrap, MetricChained):
    """Jaccard complement tn / (tn + fp + fn)"""
    name = 'jaccard_complement'

    @classmethod
    def metric_function(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        s = cls.chain(s, ConfusionMatrix, *args, **kwargs)
        s.update_metric_value(cls.name, s.get_metric_value('tn') / (s.get_metric_value('tn') + s.get_metric_value('fp') + s.get_metric_value('fn')))
        return s


class PositiveLikelihoodRatio(MetricSingleSlice, MetricCI_bootstrap, MetricChained):
    """Positive Likelihood Ratio LR+"""
    name = 'LR+'

    @classmethod
    def metric_function(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        s = cls.chain(s, TPR, *args, **kwargs)
        s = cls.chain(s, FPR, *args, **kwargs)
        s.update_metric_value(cls.name, s.get_metric_value(TPR.name) / s.get_metric_value(FPR.name))
        return s


class NegativeLikelihoodRatio(MetricSingleSlice, MetricCI_bootstrap, MetricChained):
    """Negative Likelihood Ratio LR-"""
    name = 'LR-'

    @classmethod
    def metric_function(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        s = cls.chain(s, FNR, *args, **kwargs)
        s = cls.chain(s, TNR, *args, **kwargs)
        s.update_metric_value(cls.name, s.get_metric_value(FNR.name) / s.get_metric_value(TNR.name))
        return s


class DiagnosticOddsRatio(MetricSingleSlice, MetricCI_bootstrap, MetricChained):
    """DiagnosticOddsRatio"""
    name = 'DOR'

    @classmethod
    def metric_function(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        s = cls.chain(s, PositiveLikelihoodRatio, *args, **kwargs)
        s = cls.chain(s, NegativeLikelihoodRatio, *args, **kwargs)
        s.update_metric_value(cls.name, s.get_metric_value(PositiveLikelihoodRatio.name) / s.get_metric_value(NegativeLikelihoodRatio.name))
        return s


class Prevalence(MetricSingleSlice, MetricCI_clopper_pearson, MetricChained):
    """Prevalence p / p + n"""
    name = 'prevalence'

    @classmethod
    def metric_function(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        s = cls.chain(s, ConfusionMatrix, *args, **kwargs)
        s.update_metric_value(cls.name, s.get_metric_value('p') / s.get_metric_value('size'))
        return s

    @classmethod
    def calc_CI(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        return super().calc_CI(s, 'p', 'size', *args, **kwargs)


class PredictedPrevalence(MetricSingleSlice, MetricCI_clopper_pearson, MetricChained):
    """Predicted prevalence pp / p + n"""
    name = 'predicted_prevalence'

    @classmethod
    def metric_function(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        s = cls.chain(s, ConfusionMatrix, *args, **kwargs)
        s.update_metric_value(cls.name, s.get_metric_value('pp') / s.get_metric_value('size'))
        return s

    @classmethod
    def calc_CI(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        return super().calc_CI(s, 'pp', 'size', *args, **kwargs)


class PrevalenceThreshold(MetricSingleSlice, MetricCI_bootstrap, MetricChained):
    """Prevalence Threshold"""
    name = 'prevalence_threshold'

    @classmethod
    def metric_function(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        import numpy as np
        s = cls.chain(s, FPR, *args, **kwargs)
        s = cls.chain(s, TPR, *args, **kwargs)
        s.update_metric_value(cls.name, np.sqrt(s.get_metric_value(FPR.name)) / (np.sqrt(s.get_metric_value(TPR.name)) + np.sqrt(s.get_metric_value(FPR.name))))
        return s


class Accuracy(MetricSingleSlice, MetricCI_bootstrap, MetricChained):
    """Accuracy"""
    name = 'accuracy'

    @classmethod
    def metric_function(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        s = cls.chain(s, ConfusionMatrix, *args, **kwargs)
        s.update_metric_value(cls.name, (s.get_metric_value('tp') + s.get_metric_value('tn')) / (s.get_metric_value('size')))
        return s


class Jaccard(MetricSingleSlice, MetricCI_bootstrap, MetricChained):
    """Jaccard"""
    name = 'jaccard'

    @classmethod
    def metric_function(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        s = cls.chain(s, ConfusionMatrix, *args, **kwargs)
        s.update_metric_value(cls.name, s.get_metric_value('tp') / \
            (s.get_metric_value('tp') + s.get_metric_value('fn')) / (s.get_metric_value('fp')))
        return s


class BalancedAccuracy(MetricSingleSlice, MetricCI_bootstrap, MetricChained):
    """Balanced Accuracy"""
    name = 'balanced_accuracy'

    @classmethod
    def metric_function(cls, s: DataSlice, *args, **kwargs) -> DataSlice:
        s = cls.chain(s, TPR, *args, **kwargs)
        s = cls.chain(s, TNR, *args, **kwargs)
        s.update_metric_value(cls.name, 0.5 * (s.get_metric_value(TPR.name) + s.get_metric_value(TNR.name)))
        return s


class ConfusionMatrix(MetricSingleSlice):
    """Confusion Matrix"""
    name = 'ConfusionMatrix'

    @classmethod
    def metric_function(cls, s: DataSlice, Y: tuple, *args, **kwargs) -> DataSlice:
        from sklearn.metrics import confusion_matrix
        classes = cls.settings.get('evaluation').get('binary_classification').get('classes')

        tn, fp, fn, tp = confusion_matrix(*Y, labels=classes).ravel()
        s.update_metric_value('tn', tn)
        s.update_metric_value('tp', tp)
        s.update_metric_value('fn', fn)
        s.update_metric_value('fp', fp)

        s.update_metric_value('n', tn + fp)       # total number of gt negatives
        s.update_metric_value('p', tp + fn)       # total number of gt positives
        s.update_metric_value('pn', tn + fn)      # predicted negatives
        s.update_metric_value('pp', tp + fp)      # predicted positives

        s.update_metric_value('size', tn + fp + tp + fn)  # total sample size
        s.update_metric_value(cls.name, True)     # placeholder to show that the confusion matrix has been calculated
        return s


class ExplainedVariance(MetricSingleSlice, MetricCI_bootstrap):
    """Explained Variance"""
    name = 'explained_variance'

    @classmethod
    def metric_function(cls, s: DataSlice, Y: tuple, *args, **kwargs) -> DataSlice:
        from sklearn.metrics import explained_variance_score
        s.update_metric_value(cls.name, explained_variance_score(*Y))
        return s


class R2(MetricSingleSlice, MetricCI_bootstrap):
    """Explained Variance"""
    name = 'R2'

    @classmethod
    def metric_function(cls, s: DataSlice, Y: tuple, *args, **kwargs) -> DataSlice:
        from sklearn.metrics import r2_score
        s.update_metric_value(cls.name, r2_score(*Y))
        return s
