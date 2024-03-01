import os
os.environ['WHYLOGS_NO_ANALYTICS']='True'

from deepchecks.tabular.suites import full_suite
from evidently.metric_preset import (
    DataDriftPreset, DataQualityPreset, TargetDriftPreset, RegressionPreset, ClassificationPreset
)

from evidently.report import Report
from evidently.test_preset import (
    DataStabilityTestPreset, DataQualityTestPreset, 
    BinaryClassificationTestPreset, BinaryClassificationTopKTestPreset,
    DataDriftTestPreset, MulticlassClassificationTestPreset, 
    RegressionTestPreset
)

from evidently.test_suite import TestSuite
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import whylogs as why
from whylogs.viz import NotebookProfileVisualizer

from ydata_profiling import ProfileReport

from pycaret.classification import *

def main():
    # load dataset
    from pycaret.datasets import get_data
    diabetes = get_data('diabetes')

    clf1 = setup(data = diabetes, target = 'Class variable')

    # creating a model
    lr = create_model('lr')

    plot_model(lr, plot = 'auc', save = True)


if __name__ == "__main__":
    main()