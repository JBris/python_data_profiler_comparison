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


def run_why(df):
    why_results = why.log(df)
    prof_view = why_results.view()
    visualization = NotebookProfileVisualizer()
    visualization.set_profiles(target_profile_view=prof_view, reference_profile_view=prof_view)
    
    visualization.write(
        rendered_html=visualization.profile_summary(),
        html_file_name=os.getcwd() + "/why_profile_summary",
    )

    visualization.write(
        rendered_html=visualization.double_histogram(feature_name=df.columns),
        html_file_name=os.getcwd() + "/why_double_histogram",
    )

    visualization.write(
        rendered_html=visualization.feature_statistics(feature_name=df.columns),
        html_file_name=os.getcwd() + "/why_feature_statistics",
    )

def run_evidently(df):
    evidently_test_report= TestSuite(tests=[
        DataStabilityTestPreset(), DataQualityTestPreset(), BinaryClassificationTestPreset(),
        DataDriftTestPreset(), 
        MulticlassClassificationTestPreset(), RegressionTestPreset()
    ])

    evidently_test_report.run(current_data=df.iloc[:60], reference_data=df.iloc[60:], column_mapping=None)
    evidently_test_report.save_html("evidently_test_report.html")

    evidently_metric_report = Report(metrics=[
        DataDriftPreset(), DataQualityPreset(), TargetDriftPreset(), 
        RegressionPreset(), ClassificationPreset()
    ])

    evidently_metric_report.run(current_data=df.iloc[:60], reference_data=df.iloc[60:], column_mapping=None)
    evidently_metric_report.save_html("evidently_metric_report.html")

def run_deepchecks(df):
    suite = full_suite()
    train_df = train_dataset=df.iloc[:60]
    test_df = test_dataset=df.iloc[60:]

    model = LogisticRegression().fit(train_df.drop(columns="target"), train_df["target"])
    suite_result = suite.run(train_dataset=train_df, test_dataset=test_df, model=model)
    suite_result.save_as_html("deepchecks.html") 

def run_profiler(df):
    profile = ProfileReport(df, title="Profiling Report")
    profile.to_file("ydata_profiling.html")

def main():
    iris = load_iris(as_frame=True)
    df = iris["data"]
    df["target"] = iris["target"]
    df["prediction"] = iris["target"]

    run_why(df)
    run_evidently(df)
    run_deepchecks(df)
    run_profiler(df)

if __name__ == "__main__":
    main()