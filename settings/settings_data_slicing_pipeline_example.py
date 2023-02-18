
import warnings
import logging
import time
from pathlib import Path

# pipeline
from data_slicing_pipeline.data_slicing_pipeline_classification import DataSlicingPipelineClassification as Classification

# base metrics
from metrics.metric_classification import *

# additional metrics
from metrics.confidence_and_outlier_score import *
from metrics.metric_multithreshold_classification import *
from metrics.malignancy_score import MalignancyScore

# fairness metrics
from metrics.metric_group_comparison import *

# postprocessing steps
from postprocessing.add_CI_difference import AddCIDifference
from postprocessing.add_ranking_columns import AddRankingColumns
from postprocessing.add_group_comparison_stats import AddGroupComparisonStatistics

warnings.simplefilter(action='ignore')
log_folder: Path = Path.cwd() / 'logs'
log_folder.mkdir(exist_ok=True, parents=True)
logging.basicConfig(filename=log_folder / f'data_slicing_pipeline{time.strftime("%Y%m%d_%H%M")}.log',
                    level=logging.DEBUG)

# TODO: set number of tta (test-time augmentation) columns to use for confidence score
# tta_ids = list(range(10))

settings = {
    # Multiprocessing
    'mp': {
        0: {
            'mp_metrics': {
                'use_mp': False,
            },
            'mp_data_slicing': {
                'use_mp': False,
            }
        },
        1: {
            'mp_metrics': {
                'use_mp': False,
            },
            'mp_data_slicing': {
                'use_mp': False,
            }
        },
        2: {
            'mp_metrics': {
                'use_mp': False,
                'N_threads': 16,
                'chunksize': 32,
            },
            'mp_data_slicing': {
                'use_mp': False,
                'N_threads': 16,
                'chunksize': 32,
            }
        },
        3: {
            'mp_metrics': {
                'use_mp': False,
                'N_threads': 32,
                'chunksize': 32,
            },
            'mp_data_slicing': {
                'use_mp': False,
                'N_threads': 32,
                'chunksize': 32,
            }
        }
    },
    'pipeline': Classification,
    # Dataframe structure
    'dataframe': {
        'index': 'record_id',
        # Additional columns to export from the original DataFrame
        'export': [
            'record_id',
            'malignancy_score'
        ],
        'formatting': {
            'binary_classification': {
                'pred_label_input': 'malignancy_score',
                'pred_label_output': 'malignancy_label_pred',
                'gt_label_input': 'malignancy_label_gt',
                'gt_label_output': 'malignancy_label_gt',
                'classification_threshold': 0.5,
                'na_strict': False,
                'na_fill': 'unknown',

            },
            'object_detection': {},
            'segmentation': {},
            'description_separator': '  '
        }
    },
    # Slicing settings
    'slicing': {
        'data_slicing_minimum_samples': 100,
        # TODO: update these fields to the actual column names
        # Format: 'column name as defined in datafram': 'short column name to be used in the viewer'
        'meta_data_fields_of_interest': {
            'column_name_feature1': 'feature1',
            'column_name_feature2': 'feature2',
            'column_name_feature3': 'feature3',
            'column_name_feature4': 'feature4',
        }
    },
    # Evaluaton and metric settings
    'evaluation': {
        'slice_evaluation_minimum_samples': 2,  # must be 2 or higher
        'binary_classification': {
            'classes': [0, 1],
            'n_resamples': 30,
            'metrics': {
                AUC: {'CI': True},
                Precision: {'CI': True},
                Markedness: {'CI': True},
                NPV: {'CI': True},
                Specificity: {'CI': True},
                Phi: {'CI': True},
                F1: {'CI': True},
                F1Complement: {'CI': True},
                TPR: {'CI': True},
                FNR: {'CI': True},
                FPR: {'CI': True},
                Accuracy: {'CI': True},
                Jaccard: {'CI': True},
                JaccardComplement: {'CI': True},
                Recall: {'CI': True},
                BalancedAccuracy: {'CI': True},
                Informedness: {'CI': True},
                AveragePrecision: {'CI': True},
                BalancedInformednessMarkedness: {'CI': True},
                R2: {'CI': True},
                ExplainedVariance: {'CI': True},
                DiagnosticOddsRatio: {'CI': True},
                Sensitivity: {'CI': True},
                Prevalence: {'CI': True},
                PrevalenceThreshold: {'CI': True},
                PositiveLikelihoodRatio: {'CI': True},
                NegativeLikelihoodRatio: {'CI': True},
            },
            'metrics_multi_threshold': {
                # ROC: {'CI': False},
                # PRC: {'CI': False},
            },
            'metrics_group_comparison': {
                StatisticalParityDifference: {'CI': True},
                AverageOddsDifference: {'CI': True},
                DisparateImpact: {'CI': True},
                TPRDisparity: {'CI': True},
                EqualOpportunityDifference: {'CI': True}
            },
        },
        'additional_metrics': {
            MalignancyScore: {'probabilites_columns': 'malignancy_score'},
            # ConfidenceScore: {'probabilites_columns': list([f'malignancy_score_tta_{tta}' for tta in tta_ids])},
            # OutlierScore: {'probabilites_columns': 'outlier_score'},
        },
        'postprocessing': {
            AddCIDifference: {},
            AddGroupComparisonStatistics: {
                'metrics': [KolmogorovSmirnoff],
                'columns': [ConfidenceScore.name, MalignancyScore.name]
                },
            AddRankingColumns: {},
        },
    }
}
