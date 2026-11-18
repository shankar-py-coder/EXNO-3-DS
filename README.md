## EXNO-3-DS

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
df=pd.read_csv("/Encoding Data.csv")
df
```

<img width="665" height="452" alt="image" src="https://github.com/user-attachments/assets/ec25b1df-e68e-421d-adc0-ba4f46347c37" />

```
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder
pm=['Hot','Warm','Cold']
e1=OrdinalEncoder(categories=[pm])
e1.fit_transform(df[['ord_2']])
```

<img width="548" height="276" alt="image" src="https://github.com/user-attachments/assets/988f2443-23b2-4132-ae2d-61a76feab0ce" />

```
df['bo2']=e1.fit_transform(df[['ord_2']])
df
```

<img width="662" height="458" alt="image" src="https://github.com/user-attachments/assets/5af1ec7f-a914-4214-9ec2-dc24b948389b" />


```
le=LabelEncoder()
dfc=df.copy()
dfc['ord_2']=le.fit_transform(dfc['ord_2'])
dfc

```

<img width="712" height="520" alt="image" src="https://github.com/user-attachments/assets/ae133bff-4148-4b3f-ab6d-edb85c596d7d" />

```
from sklearn.preprocessing import OneHotEncoder
ohe=OneHotEncoder(sparse_output=False)
df2=df.copy()
enc=pd.DataFrame(ohe.fit_transform(df2[['nom_0']]))
df2=pd.concat([df2,enc],axis=1)
df2
```

<img width="701" height="511" alt="image" src="https://github.com/user-attachments/assets/8e4efc33-dca8-4bc7-a8e4-4fbceb6891f0" />


```
pd.get_dummies(df2,columns=["nom_0"])
```

<img width="881" height="485" alt="image" src="https://github.com/user-attachments/assets/b9d30a40-84f1-46cb-a707-c78da0df81f1" />

```
pip install --upgrade category_encoders
```

<img width="1602" height="315" alt="image" src="https://github.com/user-attachments/assets/290d1afe-24bc-49dd-b363-4e18d264ac97" />

```
from category_encoders import BinaryEncoder
df=pd.read_csv("/data.csv")
df
```

<img width="855" height="436" alt="image" src="https://github.com/user-attachments/assets/9e76dd87-3c5f-477b-97d3-fd00425a9c0f" />

```
be=BinaryEncoder()
nd=be.fit_transform(df['Ord_2'])
dfb=pd.concat([df,nd],axis=1)
dfb1=df.copy()
dfb
```

<img width="918" height="416" alt="image" src="https://github.com/user-attachments/assets/4fb8536e-5d4c-4191-ba74-0609ca11a90c" />

```
from category_encoders import TargetEncoder
te=TargetEncoder()
cc=df.copy()
new=te.fit_transform(X=cc["City"],y=cc["Target"])
cc=pd.concat([cc,new],axis=1)
cc
```

<img width="810" height="526" alt="image" src="https://github.com/user-attachments/assets/95405d2a-8588-4bb1-b790-e4f8b3a2ed4d" />


```
import pandas as pd
from scipy import stats
import numpy as np
df=pd.read_csv("/Data_to_Transform.csv")
df
```

<img width="1018" height="510" alt="image" src="https://github.com/user-attachments/assets/9859bd62-a9af-49a6-9800-84b4b69e32d9" />


```
df.skew()
```

<img width="482" height="307" alt="image" src="https://github.com/user-attachments/assets/9f86604e-d483-4186-acf5-a8ea1f582290" />


```
np.log(df["Highly Positive Skew"])
```

<img width="527" height="567" alt="image" src="https://github.com/user-attachments/assets/d557b029-f2df-4868-a168-65a9ea60cbb7" />


```
np.reciprocal(df["Moderate Positive Skew"])
```

<img width="487" height="552" alt="image" src="https://github.com/user-attachments/assets/0fbabd88-bf62-4fd5-b9a3-ad18d43238ba" />


```
np.sqrt(df["Highly Positive Skew"])
```

<img width="492" height="551" alt="image" src="https://github.com/user-attachments/assets/80fe64d1-3ff8-48f5-88d9-d28f94967ae1" />


```
np.square(df["Highly Positive Skew"])
```

<img width="472" height="560" alt="image" src="https://github.com/user-attachments/assets/26a9075e-bd5f-4e39-aae3-e38527cfeeff" />


```
df["Highly Positive Skew_boxcox"],parameters=stats.boxcox(df["Highly Positive Skew"])
df
```

<img width="1292" height="520" alt="image" src="https://github.com/user-attachments/assets/bdebe74c-14fe-4e0f-a033-87f2aeb92b59" />


```
df["Moderate Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Moderate Negative Skew"])
df
```

<img width="1610" height="506" alt="image" src="https://github.com/user-attachments/assets/13ed37bc-fb2a-49e6-85fa-dfcb14fa9b43" />


```
df.skew()
```

<img width="517" height="327" alt="image" src="https://github.com/user-attachments/assets/e3e42f26-13cc-45ef-8687-8e811d9c20b3" />


```
df["HIghly Negative Skew_yeojohnson"],parameters=stats.yeojohnson(df["Highly Negative Skew"])
df.skew()
```

<img width="622" height="370" alt="image" src="https://github.com/user-attachments/assets/f3d63b1e-b195-4c67-b69a-cd317c1e587d" />


```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal')
df["Moderate Negative Skew_1"]=qt.fit_transform(df[["Moderate Negative Skew"]])
df
```

<img width="1672" height="537" alt="image" src="https://github.com/user-attachments/assets/6228c4d7-6e5a-4018-8615-6f7811af53ce" />


```
import seaborn as sns
import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="767" height="631" alt="image" src="https://github.com/user-attachments/assets/c9aa7e55-a301-4332-b410-379c0d574012" />


```
sm.qqplot(np.reciprocal(df["Moderate Negative Skew"]),line='45')
plt.show()
```

<img width="732" height="553" alt="image" src="https://github.com/user-attachments/assets/04702a05-c7a6-4966-b053-4b0355e07545" />


```
from sklearn.preprocessing import QuantileTransformer
qt=QuantileTransformer(output_distribution='normal',n_quantiles=891)
df["Moderate Negative Skew"]=qt.fit_transform(df[["Moderate Negative Skew"]])
sm.qqplot(df["Moderate Negative Skew"],line='45')
plt.show()
```

<img width="740" height="551" alt="image" src="https://github.com/user-attachments/assets/7339d2f4-0e2c-4029-bb57-b67e272c3941" />


```
df["Highly Negative Skew_1"]=qt.fit_transform(df[["Highly Negative Skew"]])
sm.qqplot(df['Highly Negative Skew'],line='45')
plt.show()
```

<img width="707" height="542" alt="image" src="https://github.com/user-attachments/assets/d7e819d8-3943-4d4f-a4f7-9b45bc836625" />


```
sm.qqplot(df["Highly Negative Skew_1"],line='45')
plt.show()
```

<img width="707" height="533" alt="image" src="https://github.com/user-attachments/assets/953fac75-f64d-4e37-af26-ffedd8ebae32" />

# RESULT:
Thus the python program to read the given data and perform Feature Encoding and Transformation process and save the data to a file is implemented.

       
