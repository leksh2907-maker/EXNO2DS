# EXNO2DS
```
Name: Lekshmeendhra S
reg no: 212225040198
```
# AIM:
      To perform Exploratory Data Analysis on the given data set.
      
# EXPLANATION:
  The primary aim with exploratory analysis is to examine the data for distribution, outliers and anomalies to direct specific testing of your hypothesis.
  
# ALGORITHM:
STEP 1: Import the required packages to perform Data Cleansing,Removing Outliers and Exploratory Data Analysis.

STEP 2: Replace the null value using any one of the method from mode,median and mean based on the dataset available.

STEP 3: Use boxplot method to analyze the outliers of the given dataset.

STEP 4: Remove the outliers using Inter Quantile Range method.

STEP 5: Use Countplot method to analyze in a graphical method for categorical data.

STEP 6: Use displot method to represent the univariate distribution of data.

STEP 7: Use cross tabulation method to quantitatively analyze the relationship between multiple variables.

STEP 8: Use heatmap method of representation to show relationships between two variables, one plotted on each axis.

## CODING AND OUTPUT
```
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

dt = pd.read_csv("/content/titanic_dataset.csv")
dt
```
<img width="932" height="411" alt="image" src="https://github.com/user-attachments/assets/e32ba4fa-afb0-4ce4-b272-63a4459b6b41" />


```
dt.info()
```
<img width="338" height="322" alt="image" src="https://github.com/user-attachments/assets/2b438e5e-74a8-4f75-8974-2a8f7c18003b" />

```
dt.shape
```
<img width="648" height="32" alt="image" src="https://github.com/user-attachments/assets/a5cbd3d6-dd63-46c1-8a1d-f099eac48ba1" />

```
dt.set_index("PassengerId",inplace=True)
dt
```
<img width="1025" height="515" alt="image" src="https://github.com/user-attachments/assets/912dbc1c-6476-4863-8234-0205fb7fc43c" />


```
print("Data types:")
display(dt.dtypes.to_frame("Data Type"))
```
<img width="512" height="427" alt="image" src="https://github.com/user-attachments/assets/48d1bd01-b84a-42e5-99cf-25519b797a0d" />

```

print("Missing values:")
missing=pd.DataFrame({
    "Missing count":dt.isnull().sum(),
    "Missing percentage":(dt.isna().mean()*100).round(2)
})
display(missing.sort_values("Missing percentage",ascending=False))
```

<img width="437" height="383" alt="image" src="https://github.com/user-attachments/assets/1c80c6dd-a61f-494d-953d-4cec82cad4cf" />

```
print("Exact duplicate rows:",dt.duplicated().sum())
print("Numeric summary")
display(dt.describe())
```
<img width="672" height="296" alt="image" src="https://github.com/user-attachments/assets/1d59d600-a754-4770-ba4c-59e36d276c44" />


```
dt.nunique()
```
<img width="258" height="393" alt="image" src="https://github.com/user-attachments/assets/9de3b427-6111-4c8b-b0c5-e2bd2f3653fc" />

```
dt["Survived"].value_counts()
```
<img width="287" height="202" alt="image" src="https://github.com/user-attachments/assets/b02af9ee-770e-483c-86ee-5df49abed05d" />

```
per=(dt["Survived"].value_counts()/dt.shape[0]*100).round(2)
per
```
<img width="216" height="172" alt="image" src="https://github.com/user-attachments/assets/51e23375-6b41-49fb-9ee5-b1544bf7eb6b" />


```
sns.countplot(data=dt, x="Survived")
```
<img width="652" height="455" alt="image" src="https://github.com/user-attachments/assets/9efbf5d7-a3bf-4188-ac7e-6e11904bb49a" />

```
dt.Pclass.unique()
```
<img width="1023" height="72" alt="image" src="https://github.com/user-attachments/assets/c260e161-ccf2-4e56-95df-b157f46caa0b" />

```
dt.rename(columns={'Sex':'Gender'},inplace=True) 
dt
```
<img width="1026" height="506" alt="image" src="https://github.com/user-attachments/assets/81fd6cc3-79fa-47ba-8694-a19e65da1f9a" />

```
sns.catplot(x='Gender',col='Survived',kind='count',data=dt,height=5,aspect=.7)
```
<img width="967" height="486" alt="image" src="https://github.com/user-attachments/assets/692b2386-698d-4928-89f2-eb7c6bbf46fd" />

```
sns.catplot(x='Survived', hue="Gender", data=dt, kind = "count")
```
<img width="837" height="451" alt="image" src="https://github.com/user-attachments/assets/a3972279-b27f-4a98-a77f-38004ef385de" />

```
dt.boxplot(column='Age',by='Survived')
```
<img width="732" height="450" alt="image" src="https://github.com/user-attachments/assets/39dcb1ce-1b9a-4a90-9919-49051f63d94a" />

```
sns.scatterplot(x=dt['Age'],y=dt['Fare'])

```
<img width="852" height="422" alt="image" src="https://github.com/user-attachments/assets/379e6841-4e8d-4b7a-bd25-a447517bbc83" />

```
sns.jointplot(x="Age",y="Fare", data=dt)
```
<img width="692" height="567" alt="image" src="https://github.com/user-attachments/assets/e867ee24-f615-4949-9339-cbb0bdf84e60" />

```
fig, ax1= plt.subplots(figsize=(8,5))
pt=sns.boxplot(ax=ax1,x="Pclass",y='Age', hue='Gender', data=dt)
```
<img width="817" height="422" alt="image" src="https://github.com/user-attachments/assets/023e8c35-9f48-44a0-b567-a821d6768abd" />

```
sns.catplot(data=dt, col ="Survived", x = "Gender", hue="Pclass", kind = "count")
```
<img width="1028" height="485" alt="image" src="https://github.com/user-attachments/assets/a69a8506-be01-4aed-a633-2127b60c136a" />

```
corr=dt.corr(numeric_only=True)

sns.heatmap(corr,annot=True,cmap='coolwarm')
```


<img width="682" height="412" alt="image" src="https://github.com/user-attachments/assets/755f8dcd-b134-404a-95ed-b3281218f046" />















    
                     







# RESULT
        The Program and The Output has been sucessfully Verified along with the Visualization .
