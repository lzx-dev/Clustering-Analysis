# Python Library Evaluation: AIF360

AIF360 Developed by IBM, AI Fairness 360 (AIF360)
is an extensible open source toolkit for detecting, understanding, and mitigating algorithmic biases. IBM notes
that the toolkit should only be used in a very limited setting: allocation or risk assessment problems with well-defined
protected attributes

This report will go through AIF360 setup, dataset, bias detectos, algorithms, metric, Scikit-learn compatible version, tutorial videos and community. I evaluated this libarary based on four criteria from the user's perspective: <br>

1.The abilities to use the toolkit to learn more about ML fairness and the landscape of current ML fairness research <br>
2.Toolkits should be rapidly on-boarding due to workplace time constraints <br>
3.The abilities to integrate the toolkits into existing ML working pipeline <br>
4.Using toolkits as code repositories to implement state-of-the-art or domain specific ML fairness algorithms. <br>




## 1 Setup
Conda is recommended for all configurations though Virtualenv, to create a new Python 3.7 environment, run:
```python
conda create --name aif360 python=3.7
conda activate aif360
```
#### 1.1 Pip Installation
To install the latest stable version from PyPI, run:
```python
pip install aif360
#Some algorithms require additional dependencies, for complete functionality, run:
pip install 'aif360[all]'
```
#### 1.2 Manual Installation

```python
#Clone the latest version of this repository
git clone https://github.com/Trusted-AI/AIF360
#download the datasets in https://github.com/Trusted-AI/AIF360/tree/master/aif360/data
#place them in their respective folders. Then, navigate to the #root directory of the project and run:
pip install --editable '.[all]'
```
#### 1.3 Evaluation
AIF360 provided 8 datasets in pre-defined class type for users to import, however, users must go through manual Installation to import them which reuqires open multiple links to download original files. Most tutorial notebooks provided in AIF360 github are also using pre-defined datasets. <br>
Users prefer to use tutorial notebooks to learn toolkits and they want an easy and fast installation of toolkits . In AIF360, getting access of running tutorial notebooks by manual Installation would be a problem here.


## 2 Dataset
#### 2.1 Datasets provided out of the box by aif360
AIF360 defined its own data class type.
 * datasets.StructuredDataset is base class for all structured datasets.
 * datasets.BinaryLabelDataset is base class for all structured datasets with binary labels.
 * datasets.StandardDataset is base class for every BinaryLabelDataset provided out of the box by aif360. 
 * datasets.RegressionDataset is base class for regression datasets.
```python
##All datasets provided
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

#### 2.2 Transform external data source to defined class
* Inherit StandardDataset<br>
This class will go through a standard preprocessing routine:
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

* Inherit BinaryLabelDataset <br>
```python
#To inherit Binarylabeldataset, we need to preprocess external dataset before initilize the class
ad_binary = BinaryLabelDataset(df= ad_conversion_dataset, 
                               label_names = ['true_conversion'],
                               protected_attribute_names = ['homeowner'],
                               favorable_label = 1, unfavorable_label = 0)
```
#### 2.3 Evaluation
Datasets provided by AIF360 have defined which attributes are protected and what values privileged group or unprivileged group contains, but in the libarary documentation, there is not detailed information about how the data was collected and how the features were being defined. <br>
Context documents like Datasheets could better scaffold an user’s process of issue discovery, understanding, and
ethical decision-making around ML training datasets.


## 3 Detectors
#### 3.1 Bias detector
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
#### 3.2 MDSS metric
A Metric for the bias scan scoring and scanning methods that uses the ClassificationMetric abstraction.
```python
from aif360.metrics.mdss_classification_metric import MDSSClassificationMetric 
```

```diff
-Note: import detectors or MDSS metric will raise error
```
## 4 Algorithms
#### 4.1 Apply classification Algorithms to BinaryLabel Dataset
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

```diff
-Note: import postprocessing.reject_option_classification will raise error
```
#### 4.2 Apply Regression Algorithms to Regression Dataset
```python
from aif360.algorithms.inprocessing.grid_search_reduction import GridSearchReduction
from sklearn.linear_model import LinearRegression
estimator = LinearRegression()
grid_search_red = GridSearchReduction(prot_attr="attr_name", 
                                      estimator=estimator, 
                                      constraints="DemographicParity")
grid_search_red.fit(X_train, y_train)
```

#### 4.3 evaluation
There are 14 algorithms in total, only one can be applied to regression dataset. It is also hard to apply AIF360 to models which are not binary classification; e.g., for regression models, there is not much guidance on if or how the toolkit should be used. <br>

Another problem here is redundent input. In many algorithms, they require input parameters contain dictioanry of privilleged group, unprivilleged group and sensitive attributs. If we apply these algorithms to standard dataset, which already defined what is privilleged group, unprivilleged group and sensitive attributs inside the class, then there will be repeated input. 


## 5 Metric
#### 5.1 metric for binary label dataset
```python
from aif360.metrics import BinaryLabelDatasetMetric

privileged_groups = [{"attribute_name": privileged value}]
unprivileged_groups = [{"attribute_name": unprevileged value}]

metric = BinaryLabelDatasetMetric(binary_dataset, 
                          unprivileged_groups=unprivileged_groups, 
                          privileged_groups=privileged_groups)                      
metric.disparate_impact()
```
#### 5.2 metric for computing based on two BinaryLabelDatasets
```python
from aif360.metrics import ClassificationMetric

metric = ClassificationMetric(binary_dataset, binary_dataset_pred, 
                      unprivileged_groups=unprivileged_groups,
                      privileged_groups=privileged_groups )            
metric.average_odds_difference()
metric.accuracy()
```
#### 5.3 metrics for computing based on two StructuredDatasets.
```python
from aif360.metrics.sample_distortion_metric import SampleDistortionMetric
metric = SampleDistortionMetric(structure_dataset, distorted_dataset,
                                  unprivileged_groups=unprivileged_groups,
                                  privileged_groups=privileged_groups)
metric.average_euclidean_distance()
metric.mean_euclidean_distance_difference()
```

#### 5.4 evaluation
      


## 6 Expainer
Class for explaining metric values with text
```python
MetricTextExplainer(metric).disparate_impact()
```


## 7 Scikit-learn compatible version
The dataset format for aif360.sklearn is a pandas.DataFrame with protected attributes in the index.
##### dataset 
- aif360.sklearn.datasets.standardize_dataset

separate data, targets, and possibly sample weights and populate protected attributes as sample properties
```python
import pandas as pd
from aif360.sklearn.datasets.utils import standardize_dataset
df = pd.DataFrame([[0.5, 1, 1, 0.75], [-0.5, 0, 0, 0.25]],
                   columns=['X', 'y', 'Z', 'w'])
x,y  = standardize_dataset(df, prot_attr='Z', target='y')
## note: x and y are dataframe with protected attribute in index
```
- load dataset from aif360
```python
from aif360.sklearn.datasets.openml_datasets import fetch_adult
data = fetch_adult()
#Tuple containing X, y, and sample_weights for the Adult dataset accessible by index or name.
x = data[0]
y = data[1]
sample_weight = data[2]
```
#### fairness metric
```python
#aif360.sklearn.metrics.statistical_parity_difference
#(y_true, y_pred=None, *, prot_attr=None, priv_group=1, pos_label=1, sample_weight=None)
from aif360.sklearn.metrics import statistical_parity_difference

## if prot_attr is none, all protected attributes in y_true are used
statistical_parity_difference(y_true, y_pred)
## return float: statistical parity difference

```



#### algorithm
```python
from aif360.sklearn.preprocessing.reweighing import Rewighing
## default input is none, meaning all protected attributes from the dataset are used
rw = Reweighing(["prot_attr_name"])
rw.fit_transform(x, y)
```

## 8 Tutorial Videos&Community


## Problems
* uncoordinate

In data type initilization, standardDataset label name parameter is string: Name of the label column in df. BinaryLabelDataset label name parameter is (list(str)): Names describing each label.
Favorbale class

* redundent information input 

privillaged and unprivillaged information needs to be put in both standardDataset and processing algorithm/metric

when using standardDataset, set privillaged and unprivillaged as default would be better




 
