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
```python
import pandas as pd
from aif360.datasets import StandardDataset
## import external data
ad_conversion_dataset = pd.read_csv('ad_campaign_data.csv')
## transform to StandardDataset type
ad_standard_dataset_pred = StandardDataset(df=ad_conversion_dataset, label_name= 'predicted_conversion', favorable_classes= [1],
                            scores_name= 'predicted_probability', protected_attribute_names= ['homeowner'], privileged_classes= [[0]] ,
                            categorical_features= ['parents','gender', 'college_educated','area','income', 'age'], features_to_keep=    
                            ['gender', 'age', 'income', 'area', 'college_educated', 'homeowner','parents', 'predicted_probability'])
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
from aif360.metrics import MDSSClassificationMetric
```

Note: detectors function not available in current api version

## Algorithms



## Metric





## Scikit-learn compatible


## Problems
1. Import Error
2. 
