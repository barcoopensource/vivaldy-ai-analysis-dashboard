import time
from pathlib import Path
import logging
DEBUG = False
logging_level = logging.DEBUG if DEBUG else logging.INFO
log_folder: Path = Path.cwd() / 'logs'
log_folder.mkdir(exist_ok=True, parents=True)
logging.basicConfig(filename=log_folder / f'app_dashboard{time.strftime("%Y%m%d_%H%M")}.log',
                    level=logging_level)

# metric imports for name lookup
from metrics.malignancy_score import *
from metrics.metric_classification import *
from metrics.confidence_and_outlier_score import *
from metrics.metric_group_comparison import *

TITLE = "Vivaldy Dashboard"
port_number = 8050

LAYOUT_SETTINGS = {
    'main-title': 'Vivaldy AI Analysis Dashboard',
    'primary-logo': '',
    'secondary-logo': '',
    'tertiary-logo': '',
    'array-view': {
        'x_label_left': {
            'xref': "paper",
            'yref': "paper",
            'x': -0.065,
            'showarrow': False,
            'xanchor': "left",
            'yanchor': "middle",
            'textangle': -90,
            'bgcolor': "lightgray",
            'font': {'size': 20}
            },
        'x_label_right': {
            'xref': "paper",
            'yref': "paper",
            'x': 1.05,
            'showarrow': False,
            'xanchor': "right",
            'yanchor': "middle",
            'textangle': 90,
            'bgcolor': "lightgray",
            'font': {'size': 20}
            },
        'y_label_bottom': {
            'xref': "paper",
            'yref': "paper",
            'x': 0.5,
            'y': -0.06,
            'showarrow': False,
            'xanchor': "center",
            'yanchor': "bottom",
            'font': {'size': 20}
            }
        }
}

df_file_suffix = '_sorted'


def initialize_tabs():
    import engine.widgets as w

    w.PerformanceTab()
    w.SamplesTab()
    w.FilterTab()
    w.MultithresholdCurveTab()


def initialize_widgets():
    import engine.widgets as w

    w.ModelSelector()
    w.DegreeSelector()
    w.SliceSelector()
    w.SliceLabelRankingSelector()
    w.MetricSelector()
    w.TabsBar()
    w.TabsContent()
    w.Legend()
    w.SizeThreshold()
    w.SliceLabelValueFilter()
    w.SortLabelProperty()
    w.SortLabelOperator()
    w.AxisLabelSelector()
    w.GraphTypeSelector()
    w.SortModelSelector()
    w.ExportView()

# Define model class reference
from engine.models import Model
MODEL_CLASS = Model

INITIAL_SELECTED_DEGREE = 1
INITIAL_SELECTED_METRIC = Sensitivity.name
NB_DEGREES = 3
NB_MODELS = 2

Y_PERFORMANCE_COLORS = ["CornflowerBlue", "darkred"]
Y_DETAILLED_COLORS = ["green", "olive", "red", "darkred"]
REFERENCE_SHARD_COLOR = ["lightgreen", "yellow", "pink"]


MODEL_ROOT_FOLDER = Path(__file__).parent / 'models'
ASSETS_FOLDER = Path(__file__).parent / 'assets'
EXPORTS_FOLDER = Path(__file__).parent / 'exports'

SORT_ID = 'sort_on_'

# columns on which can be sorted, but are not automatically detected as a metric or slice label
EXTRA_SORTING_COLUMN = [
    'size'
]

# Ranking options
# Define which metric should be automatically shown when a ranking option is selected
RANKING_OPTIONS = {'sensitivity': Sensitivity.name,
                   'specificity': Specificity.name,
                   'balanced': BalancedInformednessMarkedness.name,
                   'sensitivity_impact': Sensitivity.name,
                   'specificity_impact': Specificity.name,
                   'sensitivity_impact_comparison': Sensitivity.name,
                   'specificity_impact_comparison': Specificity.name,
                   }

HIGHLIGHT_N_TOP_RANKED = 10

DEFAULT_SORT_COLUMN = Sensitivity.name

# Metric settings
METRIC_OPTIONS = {
    Sensitivity.name: {'column_name': Sensitivity.name,
                       'display_name': 'Sensitivity',
                       'ylim': [0, 1.1],
                       'legend': 'Sensitivity'},
    AUC.name: {'column_name': AUC.name,
               'display_name': 'Area Under the Curve',
               'ylim': [0.0, 1.1],
               'legend': 'AUC'},
    Precision.name: {'column_name': Precision.name,
                     'display_name': 'Precision',
                     'ylim': [0, 1.1],
                     'legend': 'Precision'},
    Specificity.name: {'column_name': Specificity.name,
                       'display_name': 'Specificity',
                       'ylim': [0, 1.1],
                       'legend': 'Specificity'},
    F1.name: {'column_name': F1.name,
              'display_name': 'f1 malignant',
              'ylim': [0, 1.1],
              'legend': 'F1 Malignant'},
    F1Complement.name: {'column_name': F1Complement.name,
                        'display_name': 'f1 benign',
                        'ylim': [0, 1.1],
                        'legend': 'F1 Benign'},
    BalancedAccuracy.name: {'column_name': BalancedAccuracy.name,
                            'display_name': 'Balanced Accuracy Score',
                            'legend': 'Balanced accuracy'},
    ExplainedVariance.name: {'column_name': ExplainedVariance.name,
                             'display_name': 'Explained Variance Score',
                             'legend': 'Explained Variance Score'},
    R2.name: {'column_name': R2.name,
              'display_name': 'R2 Score',
              'legend': 'R2 Score'},
    Phi.name: {'column_name': Phi.name,
               'display_name': 'Matthews correlation coefficient',
               'legend': 'Phi'},
    DiagnosticOddsRatio.name: {'column_name': DiagnosticOddsRatio.name,
                               'display_name': 'Diagnostic Odds Ratio',
                               'legend': 'DOR'},
    Informedness.name: {'column_name': Informedness.name,
                        'display_name': 'Informedness',
                        'legend': 'Informedness'},
    Markedness.name: {'column_name': Markedness.name,
                      'display_name': 'Markedness',
                      'legend': 'Markedness'},
    BalancedInformednessMarkedness.name: {'column_name': BalancedInformednessMarkedness.name,
                                          'display_name': 'Balanced Markedness & Informedness',
                                          'legend': 'Balanced informedness/markedness'},
    ConfidenceScore.name: {'column_name': ConfidenceScore.name,
                           'display_name': 'Confidence Score',
                           'ylim': [0, 15],
                           'legend': 'Confidence'},
    OutlierScore.name: {'column_name': OutlierScore.name,
                                         'display_name': 'Outlier Score: Quantile Transform',
                                         'ylim': [-5.5, 5.5],
                                         'legend': 'Outlier QT'},
    MalignancyScore.name: {'column_name': MalignancyScore.name,
                           'display_name': 'Malignancy Score',
                           'ylim': [-0.1, 1.1],
                           'legend': 'Malignancy Score'},
    Prevalence.name: {'column_name': Prevalence.name,
                      'display_name': 'Prevalence',
                      'ylim': [-0.1, 1.1],
                      'legend': 'Prevalence'},
    PrevalenceThreshold.name: {'column_name': PrevalenceThreshold.name,
                               'display_name': 'Prevalence Threshold',
                               'ylim': [-0.1, 1.1],
                               'legend': 'Prevalence Threshold'},
    PositiveLikelihoodRatio.name: {'column_name': PositiveLikelihoodRatio.name,
                                   'display_name': 'Positive Likelyhod Ratio',
                                   'ylim': [-0.1, 20],
                                   'legend': '+ Likelihod Ratio'},
    NegativeLikelihoodRatio.name: {'column_name': NegativeLikelihoodRatio.name,
                                   'display_name': 'Negative Likelyhod Ratio',
                                   'ylim': [-0.1, 10],
                                   'legend': '- Likelihod Ratio'},
    AveragePrecision.name: {'column_name': AveragePrecision.name,
                            'display_name': 'Average Precision',
                            'ylim': [-0.1, 1.1],
                            'legend': 'Average Precision'},
    Jaccard.name: {'column_name': Jaccard.name,
                   'display_name': 'Jaccard',
                   'ylim': [-0.1, 1.1],
                   'legend': 'Jaccard'},
    JaccardComplement.name: {'column_name': JaccardComplement.name,
                             'display_name': 'JaccardComplement',
                             'ylim': [-0.1, 1.1],
                             'legend': 'JaccardComplement'},
    Accuracy.name: {'column_name': Accuracy.name,
                    'display_name': 'Accuracy',
                    'ylim': [-0.1, 1.1],
                    'legend': 'Accuracy'},

    StatisticalParityDifference.name: {'column_name': StatisticalParityDifference.name,
                    'display_name': StatisticalParityDifference.name,
                    'ylim': [-0.1, 1.1],
                    'legend': StatisticalParityDifference.name},
    DisparateImpact.name: {'column_name': DisparateImpact.name,
                    'display_name': DisparateImpact.name,
                    'legend': DisparateImpact.name},
    TPRDisparity.name: {'column_name': TPRDisparity.name,
                    'display_name': TPRDisparity.name,
                    'legend': TPRDisparity.name},


    f'{KolmogorovSmirnoff.name}_{ConfidenceScore.name}_statistic': {'column_name': f'{KolmogorovSmirnoff.name}_{ConfidenceScore.name}_statistic',
                    'display_name': f'{KolmogorovSmirnoff.name}_{ConfidenceScore.name}_statistic',
                    'legend': f'KolmogorovSmirnov_{ConfidenceScore.name}'},

    f'{KolmogorovSmirnoff.name}_{MalignancyScore.name}_statistic': {'column_name': f'{KolmogorovSmirnoff.name}_{MalignancyScore.name}_statistic',
                    'display_name': f'{KolmogorovSmirnoff.name}_{MalignancyScore.name}_statistic',
                    'legend': f'KolmogorovSmirnov_{MalignancyScore.name}'},

}
DEFAULT_METRIC = next(iter(METRIC_OPTIONS))

# Violin plot settings
BOX_PLOT_METRICS = [ConfidenceScore.name,
                    OutlierScore.name,
                    MalignancyScore.name]
SHOW_BOX_PLOT = False

# Samples per Slice settings
DETAILLED_BAR_PLOT = True
DETAILLED_BAR_PLOT_ORDER = ["tn", "tp", "fn", "fp"]

# ROC Performance per Slice settings
ROC_PERFORMANCE = {'show_CI': True,
                   'allow_CI_bootstrapping_fallback': True,
                   'CI_resamples': 100}
