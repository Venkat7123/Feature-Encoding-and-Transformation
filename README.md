## FEATURE ENCODING AND TRANSFORMATION

# AIM:
To read the given data and perform Feature Encoding and Transformation process and save the data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Encoding for the feature in the data set.
STEP 4:Apply Feature Transformation for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE ENCODING:
1. Ordinal Encoding
An ordinal encoding involves mapping each unique label to an integer value. This type of encoding is really only appropriate if there is a known relationship between the categories. This relationship does exist for some of the variables in our dataset, and ideally, this should be harnessed when preparing the data.
2. Label Encoding
Label encoding is a simple and straight forward approach. This converts each value in a categorical column into a numerical value. Each value in a categorical column is called Label.
3. Binary Encoding
Binary encoding converts a category into binary digits. Each binary digit creates one feature column. If there are n unique categories, then binary encoding results in the only log(base 2)ⁿ features.
4. One Hot Encoding
We use this categorical data encoding technique when the features are nominal(do not have any order). In one hot encoding, for each level of a categorical feature, we create a new variable. Each category is mapped with a binary variable containing either 0 or 1. Here, 0 represents the absence, and 1 represents the presence of that category.

# Methods Used for Data Transformation:
  # 1. FUNCTION TRANSFORMATION
• Log Transformation
• Reciprocal Transformation
• Square Root Transformation
• Square Transformation
  # 2. POWER TRANSFORMATION
• Boxcox method
• Yeojohnson method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
from scipy import stats
df = pd.read_csv('/content/Encoding Data (2).csv')
df
```
![alt text](<Output Screenshots/Screenshot 2025-05-03 100059.png>)
```
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
pm = ['Hot', 'Warm', 'Cold']
el = OrdinalEncoder(categories=[pm])
el.fit_transform(df[['ord_2']])
```
![alt text](<Output Screenshots/Screenshot 2025-05-03 100123.png>)
```
df['bo2'] = el.fit_transform(df[['ord_2']])
df
```
![alt text](<Output Screenshots/Screenshot 2025-05-03 100130.png>)
```
le = LabelEncoder()
dfc = df.copy()
dfc['ord_2'] = le.fit_transform(dfc['ord_2'])
dfc
```
![alt text](<Output Screenshots/Screenshot 2025-05-03 100149.png>)
```
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse_output=False)
df2 = df.copy
enc = pd.DataFrame(ohe.fit_transform(df[['nom_0']]))
df2 = pd.concat([df, enc], axis=1)
pd.get_dummies(df2,columns=['nom_0'])
```
![alt text](<Output Screenshots/Screenshot 2025-05-03 100202.png>)
```
pip install --upgrade category_encoders
```
![alt text](<Output Screenshots/Screenshot 2025-05-03 100216.png>)
```
from category_encoders import BinaryEncoder
dt = pd.read_csv('/content/data (2).csv')
be = BinaryEncoder()
nd = be.fit_transform(dt['Ord_2'])
dfb = pd.concat([dt,nd],axis=1)
dfb1 = df.copy()
dfb
```
![alt text](<Output Screenshots/Screenshot 2025-05-03 100251.png>)
```
from category_encoders import TargetEncoder
te = TargetEncoder()
cc = dfb.copy()
new = te.fit_transform(X= cc['City'], y=cc['Target'])
cc = pd.concat([cc,new],axis=1)
cc
```
![alt text](<Output Screenshots/Screenshot 2025-05-03 100305.png>)
```
dfd = pd.read_csv('/content/Data_to_Transform (1).csv')
dfd
```
![alt text](<Output Screenshots/Screenshot 2025-05-03 100320.png>)
```
dfd.skew()
```
![alt text](<Output Screenshots/Screenshot 2025-05-03 100336.png>)
```
np.log(dfd['Highly Positive Skew'])
```
![alt text](<Output Screenshots/Screenshot 2025-05-03 100345.png>)
```
np.reciprocal(dfd["Moderate Positive Skew"])
```
![alt text](<Output Screenshots/Screenshot 2025-05-03 100357.png>)
```
np.sqrt(dfd['Highly Positive Skew'])
```
![alt text](<Output Screenshots/Screenshot 2025-05-03 100411.png>)
```
np.square(dfd['Highly Positive Skew'])
```
![alt text](<Output Screenshots/Screenshot 2025-05-03 100441.png>)
```
dfd['Highly Positive Skew_boxcox'] , parameters = stats.boxcox(dfd['Highly Positive Skew'])
dfd
```
![alt text](<Output Screenshots/Screenshot 2025-05-03 100515.png>)
```
dfd['Moderate Negative Skew_yeojohnson'] , parameters = stats.yeojohnson(dfd['Moderate Negative Skew'])
dfd.skew()
```
![alt text](<Output Screenshots/Screenshot 2025-05-03 100522.png>)
```
dfd['Highly Negative Skew_yeojohnson'], parameters = stats.yeojohnson(dfd['Highly Negative Skew'])
dfd.skew()
```
![alt text](<Output Screenshots/Screenshot 2025-05-03 100527.png>)
```
from sklearn.preprocessing import QuantileTransformer
qt = QuantileTransformer(output_distribution='normal')
dfd['Moderate Negative Skew_1'] = qt.fit_transform(dfd[['Moderate Negative Skew']])
dfd
```
![alt text](<Output Screenshots/Screenshot 2025-05-03 100541.png>)
```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(dfd['Highly Negative Skew'], line='45')
plt.show()
```
![alt text](<Output Screenshots/Screenshot 2025-05-03 100550.png>)
```
sm.qqplot(np.reciprocal(dfd["Moderate Negative Skew"]),line = '45')
plt.show()
```
![alt text](<Output Screenshots/Screenshot 2025-05-03 100602.png>)
```
qt = QuantileTransformer(output_distribution='normal', n_quantiles=891)
dfd['Moderate Negative Skew'] = qt.fit_transform(dfd[['Moderate Negative Skew']])
sm.qqplot(dfd['Moderate Negative Skew'], line='45')
plt.show()
```
![alt text](<Output Screenshots/Screenshot 2025-05-03 100609.png>)
```
dfd['Highly Negative Skew_1'] = qt.fit_transform(dfd[['Highly Negative Skew']])
sm.qqplot(dfd['Highly Negative Skew'], line='45')
plt.show()
```
![alt text](<Output Screenshots/Screenshot 2025-05-03 100616.png>)
# RESULT:
Feature Encoding and Transformation was completed and the results have been successfully verified.

       
