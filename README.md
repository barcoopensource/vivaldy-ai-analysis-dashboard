# vivaldy-ai-analysis-dashboard

# Introduction
The Vivaldy (<b>V</b>er<b>I</b>fication and <b>V</b>alidation of <b>A</b>i-enab<b>L</b>e<b>D</b> s<b>Y</b>stems) analysis tool and dashboard allows for automated subgroup performance analysis for AI algorithms.

Guaranteeing the safety and effectivness in CAD-AI for all subpopulations is a key challenge. This should be verified both during initial data collection and development phase, but also during the regulatory application process, when an update of the AI is planned, and during post-deployment where both labeled and unlabeled (i.e. with absence of ground truth knowledge) field data requires monitoring for potential data and usage drift.

To make a correct impact assessment in an automated and systematic approach, supporting tools can make a difference.

Key functionalities of this repository helping with this task are:

* Take into account the subpopulation's size by using **confidence intervals**
  * Analytic formula e.g. Clopper-Pearson method for sensitivity and specificity
  * Generic bootstrapping method for complex metrics, e.g. the Matthews correlation coefficient and Diagnostic Odds Ratio
* **Fully configurable** and possible to add **custom metrics** for project-tailored pipelines
* **Bias and fairness** metrics
* Both labeled and unlabelled data, for **monitoring of field data** without known ground truth
* **Multiple ranking** methods, adaptable to the problem statement; including complex ranking methods taking into account the confidence interval, cfr. inferority/superiority statistical tests.
* Inspect **multithreshold metrics** such as ROC at subpopulation level, with CI

![figure abstract](https://github.com/barcoopensource/vivaldy-ai-analysis-dashboard/blob/main/assets/abstract.png?raw=true)


For a tutorial on the functionality, see the following video:

[![Vivaldy video tutorial](https://github.com/barcoopensource/vivaldy-ai-analysis-dashboard/blob/main/assets/video_screenshot.png?raw=true)](https://youtu.be/CxM5w6ULGq4 "Vivaldy video tutorial")

# Setup
Note: this package was tested to run with `Python 3.10` 

To install the necessary requirements, open a console and run:
```console
    pip install -r requirements.txt
```

To fetch the submodule https://github.com/yandexdataschool/roc_comparison.git run
```console
    git submodule init
```
# Usage
## Functionality
Expected input:

* Dataset with metadata labels + AI inference output
* Config file defining
    * Ground truth column name
    * AI output column name
    * Metadata column names to consider
    * Optional task-dependent values; e.g. classification threshold

Output:

* Metric values
    * Generic metrics (sensitivity, specificity, AUC, ...)
    * Custom metrics
    * Metrics for field data without known ground truth (e.g. confidence score and outlier score)
    * Bias and fairness metrics
    * Confidence intervals
* Ranked slices
    * Generic: sensitivity, specificity
    * Impactfull: taking into account the confidence intervals
    * Customizable according to research question
* Multithreshold metrics
    * ROC per subgroup



## Dashboard
To test the dashboard, the example notebooks must be ran first, which will fetch and format the required datasets.
* `example_BCSC.ipynb`
* `example_BCWD.ipynb`

To solely start the dashboard, run 
```console
    python.exe dashboard.py
```

# Examples
## BCSC - Breast Cancer Surveillance Consortium Dataset
This end-to-end example trains a binary classification network on the BCSC dataset using a 4-fold stratification. Inference results from the different models on their respective validation datasets are combined to a full dataset to be served to the Vivaldy analysis pipeline and dashboard.

During the analysis four methods are given as examples. The configurations full and minimal reflect to the amount of metrics specified. The field *fraction* corresponds to the fraction of the full dataset that will be used during analysis.

This tutorial uses data from the BCSC. The following must be cited when using this dataset:

    "Data collection and sharing was supported by the National Cancer Institute-funded Breast Cancer Surveillance Consortium (HHSN261201100031C). You can learn more about the BCSC at: http://www.bcsc-research.org/."

See notebook `example_BCSC.ipynb`

## BCWD - Breast Cancer Wisconsin Diagnostic Dataset
This end-to-end example trains a binary classification network on the BCWD dataset using a 4-fold stratification. Inference results from the different models on their respective validation datasets are combined to a full dataset to be served to the Vivaldy analysis pipeline and dashboard.

See notebook `example_BCWD.ipynb`


# More information
More information can be found in our paper published at SPIE MI 2023, San Diego
https://doi.org/10.1117/12.2653453

    Stijn Vandewiele, Jonas De Vylder, Bart Diricx, Edward Sandra, and Tom Kimpe "Open-source tool for model performance analysis for subpopulations", Proc. SPIE 12465, Medical Imaging 2023: Computer-Aided Diagnosis, 1246521 (7 April 2023)

# Acknowledgements
This rationale behind this software package was inspired by SliceFinder https://github.com/yeounoh/slicefinder

The research making this repository possible was funded through the Vivaldy project, PENTA 19021, and financially supported by the Flemish Government HBC.2019.274 https://www.vivaldy-penta.eu/