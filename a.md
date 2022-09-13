# Python Library Evaluation: AIF360

## Setup
Conda is recommended for all configurations though Virtualenv, to create a new Python 3.7 environment, run:
```python
conda create --name aif360 python=3.7
conda activate aif360
```

#### Pip Installation
To install the latest stable version from PyPI, run:
```python
pip install aif360
```
Some algorithms require additional dependencies, for complete functionality, run:
```python
pip install 'aif360[all]'
```

AIF360 provided 8 common dataset, must go through Manual Installation to import them.
#### Manual Installation
Clone the latest version of this repository
```python
git clone https://github.com/Trusted-AI/AIF360
```
download the datasets in https://github.com/Trusted-AI/AIF360/tree/master/aif360/data and place them in their respective folders. Then, navigate to the root directory of the project and run:
```python
pip install --editable '.[all]'
```

## Dataset
AIF360 defined its own data class type.
 * datasets.StructuredDataset is base class for all structured datasets.
 * datasets.BinaryLabelDataset is base class for all structured datasets with binary labels.
 * datasets.StandardDataset is base class for every BinaryLabelDataset provided out of the box by aif360. 
 * datasets.RegressionDataset is base class for regression datasets.

All datasets provided:
```python
#Regression Dataset
from aif360.datasets import LawSchoolGPADataset 
#Binary Label Dataset
from aif360.datasets import BankDataset
from aif360.datasets import CompasDataset
from aif360.datasets import AdultDataset
from aif360.datasets import GermanDataset
#Binary Label and Panel Dataset
from aif360.datasets impor MEPSDataset19
from aif360.datasets impor MEPSDataset20
from aif360.datasets impor MEPSDataset21
```

#### Transform external data source to defined class
* Inherit StandardDataset
"It is not strictly necessary to inherit StandardDataset when adding custom datasets but it may be useful."

This class will go through a standard preprocessing routine which:
- (optional) Performs some dataset-specific preprocessing (e.g. renaming columns/values, handling missing data).
- Drops unrequested columns (see features_to_keep and features_to_drop for details).
- Drops rows with NA values.
- Creates a one-hot encoding of the categorical variables.
- Maps protected attributes to binary privileged/unprivileged values (1/0).
- Maps labels to binary favorable/unfavorable labels (1/0).

```python
import pandas as pd
from aif360.datasets import StandardDataset
## import external data
ad_conversion_dataset = pd.read_csv('ad_campaign_data.csv')
## transform to StandardDataset type
ad_standard_dataset = StandardDataset(df=ad_conversion_dataset, label_name= 'true_conversion', favorable_classes= [1],
                            protected_attribute_names= ['homeowner'], privileged_classes= [[0]] ,
                            categorical_features= ['parents','gender', 'college_educated','area','income', 'age'], features_to_keep=    
                            ['gender', 'age', 'income', 'area', 'college_educated', 'homeowner','parents', 'predicted_probability'])
```

* Inherit BinaryLabelDataset
To inherit Binarylabeldataset, we need to preprocess external dataset before initilize the class
```python
ad_binary = BinaryLabelDataset(df= ad_conversion_dataset, 
                               label_names = ['true_conversion'],
                               protected_attribute_names = ['homeowner'],
                               favorable_label = 1, unfavorable_label = 0)
```

## Detectors
Multi dimensional subset scan evaluation for automated identification of subgroups that have predictive bias.
```python
from aif360.detectors.mdss.ScoringFunctions import Bernoulli
from aif360.detectors.mdss.MDSS import MDSS

scoring_function = Bernoulli(direction='negative')
scanner = MDSS(scoring_function)
scanner.scan(ad_conversion_dataset[features_4_scanning],
              expectations = ad_conversion_dataset['predicted_conversion'],
              outcomes = ad_conversion_dataset['true_conversion'],
              penalty = 1,
              num_iters = 1,
              verbose = False)

```
or
A Metric for the bias scan scoring and scanning methods that uses the ClassificationMetric abstraction.
```python
from aif360.metrics.mdss_classification_metric import MDSSClassificationMetric 
```

Note: import detectors will raise error

## Algorithms
Apply classification Algorithms to BinaryLabel Dataset
* preprocessing
```python
from aif360.algorithms.preprocessing import Reweighing
privileged_groups = [{"attribute_name": privileged value}]
unprivileged_groups = [{"attribute_name": unprevileged value}]
RW = Reweighing(unprivileged_groups=unprivileged_groups,
                privileged_groups=privileged_groups)
dataset_transf = RW.fit_transform(dataset_orig)
```
* inprocessing
```python
from aif360.algorithms.inprocessing import PrejudiceRemover
#Prejudice remover is an in-processing technique that adds 
#a discrimination-aware regularization term to the learning objective
sens_attr = "sens_attr_name"
#eta (double, optional) – fairness penalty parameter
#sensitive_attr (str, optional) – name of protected attribute
model = PrejudiceRemover(sensitive_attr=sens_attr, eta=25.0)
model.fit(dataset)
````

* postprocessing
```python
from aif360.algorithms.postprocessing import RejectOptionClassification
#Reject option classification is a postprocessing technique that gives favorable outcomes 
#to unpriviliged groups and unfavorable outcomes to priviliged groups in a confidence band around 
#the decision boundary with the highest uncertainty

# Upper and lower bound on the fairness metric used
metric_ub = 0.05
metric_lb = -0.05

ROC = RejectOptionClassification(unprivileged_groups=unprivileged_groups,
                                 privileged_groups=privileged_groups,
                                            low_class_thresh=0.01, high_class_thresh=0.99,
                                            num_class_thresh=100, num_ROC_margin=50,
                                            metric_name=metric_name,
                                            metric_ub=metric_ub, metric_lb=metric_lb)

ROC.fit(dataset_orig, dataset_pred)

ROC.classification_threshold
ROC.ROC_margin
```
The ROC method has estimated that the optimal classification threshold and the margin. This means that to mitigate bias, for instances with a predicted_probability between threshold-margin and threshold+margin, if they belong to the unprivileged group, they will be assigned a favorable outcome . However, if they belong to the privileged group, they will be assigned an unfavorable outcome.

## Metric
- metric for binary label dataset
```python
from aif360.metrics import BinaryLabelDatasetMetric

privileged_groups = [{"attribute_name": privileged value}]
unprivileged_groups = [{"attribute_name": unprevileged value}]

metric = BinaryLabelDatasetMetric(binary_dataset, 
                          unprivileged_groups=unprivileged_groups, 
                          privileged_groups=privileged_groups)                      
metric.disparate_impact()
```
- metric for computing based on two BinaryLabelDatasets
```python
from aif360.metrics import ClassificationMetric

metric = ClassificationMetric(binary_dataset, binary_dataset_pred, 
                      unprivileged_groups=unprivileged_groups,
                      privileged_groups=privileged_groups )            
metric.average_odds_difference()
metric.accuracy()
```
- metrics for computing based on two StructuredDatasets.
```python
from aif360.metrics.sample_distortion_metric import SampleDistortionMetric
metric = SampleDistortionMetric(structure_dataset, distorted_dataset,
                                  unprivileged_groups=unprivileged_groups,
                                  privileged_groups=privileged_groups)
metric.average_euclidean_distance()
metric.mean_euclidean_distance_difference()
```
      


## Expainer
Class for explaining metric values with text
```python
MetricTextExplainer(metric).disparate_impact()
```


## Scikit-learn compatible version


## Problems
* uncoordinate

In data type initilization, standardDataset label name parameter is string: Name of the label column in df. BinaryLabelDataset label name parameter is (list(str)): Names describing each label.
Favorbale class

* redundent information input 

privillaged and unprivillaged information needs to be put in both standardDataset and processing algorithm/metric

when using standardDataset, set privillaged and unprivillaged as default would be better

*  Import Error


 
