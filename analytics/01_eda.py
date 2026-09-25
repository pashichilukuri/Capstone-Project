import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV

# Classification Models & Trees
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

# Regression Models
from sklearn.linear_model import LinearRegression

# Class Imbalance Utilities
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline

# Evaluation Metrics
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             classification_report, confusion_matrix, roc_auc_score,
                             roc_curve, ConfusionMatrixDisplay, 
                             mean_absolute_error, mean_squared_error, r2_score)


df= sns.load_dataset('titanic')
df

df.to_csv("titanic.csv", index=False)
df
df.info()
df.describe()
df.shape

df_missing=(df.isnull().sum()/len(df))* 100
print(df_missing.round(2).astype(str) + "%")

#age 19.865320 % embarked 0.224467 mode deck 77.216611 mode embark_town 0.224467 mode

df["age"]=df["age"].fillna(df["age"].median())
df["age"]

before =len(df)
df= df.dropna(subset=["embarked"])
after = len(df)
print(f"droped {before-after} remaining rows where embarked is Naan/null")
print(f"remaining rows {after}")

before =len(df)
df= df.dropna(subset=["embark_town"])
after = len(df)
print(f"droped {before-after} remaining rows where embark_town is Naan/null")
print(f"remaining rows {after}")

df["deck"] = df["deck"].cat.add_categories("Missing")
df["deck"] = df["deck"].fillna("Missing")
df_missing=(df.isnull().sum()/len(df))* 100
print(df_missing.round(2).astype(str) + "%")

fig, axis = plt.subplots(2,2,figsize=(12, 8))
#hist plot for age
axis[0,0].hist(df['age'].dropna(), bins=30, color='darkblue',
         edgecolor='black', alpha=0.7)
axis[0,0].set_xlabel('age')
axis[0,0].set_ylabel('Frequency')
axis[0,0].set_title('Histogram of age')

#boxplot for age
axis[0,1].boxplot(df['age'].dropna(),vert=False)
axis[0,1].set_title('boxplot of age')
axis[0,1].set_xlabel('age')

#hist plot for fare
axis[1,0].hist(df['fare'].dropna(), bins=30, color='darkblue',
         edgecolor='black', alpha=0.7)
axis[1,0].set_xlabel('fare')
axis[1,0].set_ylabel('Frequency')
axis[1,0].set_title('Histogram of fare')

#boxplot for fare
axis[1,1].boxplot(df['fare'].dropna(),vert=False)
axis[1,1].set_title('boxplot of fare')
axis[1,1].set_xlabel('fare')
plt.tight_layout()
plt.show()
plt.savefig("univariate_age_fare.png", dpi=150)

q1_age = df["age"].quantile(0.25)
q3_age = df["age"].quantile(0.75)
IQR_age = q3_age-q1_age
print(IQR_age)
age_lower = q1_age-(1.5*IQR_age)
age_upper = q3_age+(1.5*IQR_age)
print(age_lower, age_upper)
outlier_age = df[(df["age"]<age_lower) | (df["age"]>age_upper)]
outlier_age


q1_fare= df["fare"].quantile(0.25)
q3_fare = df["fare"].quantile(0.75)
IQR_fare = q3_fare-q1_fare
print(IQR_fare)
fare_lower = q1_fare-(1.5*IQR_fare)
fare_upper = q3_fare+(1.5*IQR_fare)
print(fare_lower, fare_upper)
outlier_fare = df[(df["fare"]<fare_lower) | (df["fare"]>fare_upper)]
outlier_fare

mean=df["fare"].mean()
median=df["fare"].median()
mode = df["fare"].mode().iloc[0]

print(f" mean={mean:.2f}, median={median:.2f}, mode={mode}")

if mean > median > mode:
    print("Right-skewed (mean > median > mode)")
elif mean < median < mode:
    print("Left-skewed (mean < median < mode)")
else:
    print("Roughly symmetric / mixed ordering")
    

#using boolean masking (with &/|combinations),compute and report survival rate broken down by (a) sex, (b) pclass, and (c) sex and pclass together

#male and female passengers
male = df[df["sex"] =="male"]
female = df[df["sex"] =="female"]

#survival rate (a) sex
male_survior_rate = male['survived'].mean()
female_survior_rate = female['survived'].mean()

print(f"male_survior_rate:{male_survior_rate:.2f}, female_survior_rate:{female_survior_rate:.2f}")

#survival rate (b) pclass

pclass_1 = df[df["pclass"]==1]
pclass_2 = df[df["pclass"]==2]
pclass_3 = df[df["pclass"]==3]

pclass_1_survior_rate = pclass_1['survived'].mean()
pclass_2_survior_rate = pclass_2['survived'].mean()
pclass_3_survior_rate = pclass_3['survived'].mean()

print(f" 1st class survivior rate : {pclass_1_survior_rate:.2f}")
print(f" 2st class survivior rate : {pclass_2_survior_rate:.2f}")
print(f" 3rd class survivior rate : {pclass_3_survior_rate:.2f}")

#select exactly these six list of columns for correlation matrix
numeric_columns = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
#correlation matrix
correlation_matrix= df[numeric_columns].corr()
print(correlation_matrix.round(2))
#heat map for correlation_matrix
plt.figure(figsize=(9, 7))
sns.heatmap(correlation_matrix,square=True, cmap="coolwarm",linewidths=0.5,annot = True)
plt.savefig("correlation_heatmap.png", dpi=150)
plt.tight_layout()
plt.show()



# Columns used in the analysis
cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']

# -----------------------------------------
# Chart 1: Survival rate by sex
# -----------------------------------------
survival_by_sex = df.groupby('sex')['survived'].mean().reset_index()

plt.figure(figsize=(7, 5))
sns.barplot(data=survival_by_sex, x='sex', y='survived')
plt.title('Survival Rate by Sex')
plt.xlabel('Sex')
plt.ylabel('Survival Rate')
plt.ylim(0, 1)
plt.show()


df= sns.load_dataset('titanic')
df

df.to_csv("titanic.csv", index=False)
df

df.info()
df.describe()

df.shape

df_missing=(df.isnull().sum()/len(df))* 100
print(df_missing.round(2).astype(str) + "%")

df["age"]=df["age"].fillna(df["age"].median())
df["age"]



before =len(df)
df= df.dropna(subset=["embarked"])
after = len(df)
print(f"droped {before-after} remaining rows where embarked is Naan/null")
print(f"remaining rows {after}")



before =len(df)
df= df.dropna(subset=["embark_town"])
after = len(df)
print(f"droped {before-after} remaining rows where embark_town is Naan/null")
print(f"remaining rows {after}")

df[["age","embarked","deck","embark_town"]]



df["deck"] = df["deck"].cat.add_categories("Missing")
df["deck"] = df["deck"].fillna("Missing")



df_missing=(df.isnull().sum()/len(df))* 100
print(df_missing.round(2).astype(str) + "%")

#age 19.865320 % embarked 0.224467 mode deck 77.216611 mode embark_town 0.224467 mode

#histogram and a box plot for both age and fare.
fig, axis = plt.subplots(2,2,figsize=(12, 8))
#hist plot for age
axis[0,0].hist(df['age'].dropna(), bins=30, color='darkblue',
         edgecolor='black', alpha=0.7)
axis[0,0].set_xlabel('age')
axis[0,0].set_ylabel('Frequency')
axis[0,0].set_title('Histogram of age')

#boxplot for age
axis[0,1].boxplot(df['age'].dropna(),vert=False)
axis[0,1].set_title('boxplot of age')
axis[0,1].set_xlabel('age')

#boxplot for fare
axis[1,1].boxplot(df['fare'].dropna(),vert=False)
axis[1,1].set_title('boxplot of fare')
axis[1,1].set_xlabel('fare')
plt.tight_layout()
plt.show()
plt.savefig("univariate_age_fare.png", dpi=150)

#the IQR rule  for age (outliers are points outside [Q1 − 1.5×IQR, Q3 + 1.5×IQR])
q1_age = df["age"].quantile(0.25)
q3_age = df["age"].quantile(0.75)
IQR_age = q3_age-q1_age
print(IQR_age)

age_lower = q1_age-(1.5*IQR_age)
age_upper = q3_age+(1.5*IQR_age)
print(age_lower, age_upper)

outlier_age = df[(df["age"]<age_lower) | (df["age"]>age_upper)]
outlier_age

#the IQR rule  for fare (outliers are points outside [Q1 − 1.5×IQR, Q3 + 1.5×IQR])
q1_fare= df["fare"].quantile(0.25)
q3_fare = df["fare"].quantile(0.75)

IQR_fare = q3_fare-q1_fare
print(IQR_fare)

fare_lower = q1_fare-(1.5*IQR_fare)
fare_upper = q3_fare+(1.5*IQR_fare)
print(fare_lower, fare_upper)

outlier_fare = df[(df["fare"]<fare_lower) | (df["fare"]>fare_upper)]
outlier_fare

#calculating the mean, median, and mode of the fare column and report whether the distribution is right-skewed, left-skewed, or roughly symmetric based on the ordering of these three measures of central tendency.
mean=df["fare"].mean()
median=df["fare"].median()
mode = df["fare"].mode().iloc[0]

print(f" mean={mean:.2f}, median={median:.2f}, mode={mode}")

if mean > median > mode:
    print("Right-skewed (mean > median > mode)")
elif mean < median < mode:
    print("Left-skewed (mean < median < mode)")
else:
    print("Roughly symmetric / mixed ordering")
 #mean=32.10, median=14.45, mode=8.05
# resultRight-skewed (mean > median > mode)


#using boolean masking (with &/|combinations),compute and report survival rate broken down by (a) sex, (b) pclass, and (c) sex and pclass together

#male and female passengers
male = df[df["sex"] =="male"]
female = df[df["sex"] =="female"]

#survival rate (a) sex
male_survior_rate = male['survived'].mean()
female_survior_rate = female['survived'].mean()

print(f"male_survior_rate:{male_survior_rate:.2f}, female_survior_rate:{female_survior_rate:.2f}")

#survival rate (b) pclass

pclass_1 = df[df["pclass"]==1]
pclass_2 = df[df["pclass"]==2]
pclass_3 = df[df["pclass"]==3]

pclass_1_survior_rate = pclass_1['survived'].mean()
pclass_2_survior_rate = pclass_2['survived'].mean()
pclass_3_survior_rate = pclass_3['survived'].mean()

print(f" 1st class survivior rate : {pclass_1_survior_rate:.2f}")
print(f" 2st class survivior rate : {pclass_2_survior_rate:.2f}")
print(f" 3rd class survivior rate : {pclass_3_survior_rate:.2f}")

#Male 1st,2nd and 3rd class
Male_pclass_1 = df[(df["sex"] =="male")&(df["pclass"]==1)]
Male_pclass_2 = df[(df["sex"] =="male")&(df["pclass"]==2)]
Male_pclass_3 = df[(df["sex"] =="male")&(df["pclass"]==3)]

#Survior rate male and pclass
Male_survior_rate_pclass_1 = df[(df["sex"] =="male")&(df["pclass"]==1)&(df["survived"] == 1)]
Male_survior_rate_pclass_2 = df[(df["sex"] =="male")&(df["pclass"]==2)&(df["survived"] == 1)]
Male_survior_rate_pclass_3 = df[(df["sex"] =="male")&(df["pclass"]==3)&(df["survived"] == 1)]

#Female 1st,2nd and 3rd class
Female_pclass_1 = df[(df["sex"] =="female")&(df["pclass"]==1)]
Female_pclass_2 = df[(df["sex"] =="female")&(df["pclass"]==2)]
Female_pclass_3 = df[(df["sex"] =="female")&(df["pclass"]==3)]

##Survior rate female and pclass
Female_survior_rate_pclass_1 = df[(df["sex"] =="male")&(df["pclass"]==1)&(df["survived"] == 1)]
Female_survior_rate_pclass_2 = df[(df["sex"] =="male")&(df["pclass"]==2)&(df["survived"] == 1)]
Female_survior_rate_pclass_3 = df[(df["sex"] =="male")&(df["pclass"]==3)&(df["survived"] == 1)]

print(f"First class men survival rate: {len(Male_survior_rate_pclass_1) / len(Male_pclass_1):.1%}")
print(f"Second class men survival rate: {len(Male_survior_rate_pclass_2) / len(Male_pclass_2):.1%}")
print(f"Third class men survival rate: {len(Male_survior_rate_pclass_3) / len(Male_pclass_3):.1%}")

print(f"First class women survival rate: {len(Female_survior_rate_pclass_1) / len(Female_pclass_1):.1%}")
print(f"Second class women survival rate: {len(Female_survior_rate_pclass_2) / len(Female_pclass_2):.1%}")
print(f"Third class women survival rate: {len(Female_survior_rate_pclass_3) / len(Female_pclass_3):.1%}")

#select exactly these six list of columns for correlation matrix
numeric_columns = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
#correlation matrix
correlation_matrix= df[numeric_columns].corr()
print(correlation_matrix.round(2))

#heat map for correlation_matrix
plt.figure(figsize=(9, 7))
sns.heatmap(correlation_matrix,square=True, cmap="coolwarm",linewidths=0.5,annot = True)
plt.savefig("correlation_heatmap.png", dpi=150)
plt.tight_layout()
plt.show()

#produce at least 4 distinct charts (any combination of bar/box/scatter/heatmap/pair-plot) that together build a coherent argument about who was more likely to survive and why

# Columns used in the analysis
cols = ['survived', 'pclass', 'age', 'sibsp', 'parch', 'fare']

# -----------------------------------------
# Chart 1: Survival rate by sex
# -----------------------------------------
survival_by_sex = df.groupby('sex')['survived'].mean().reset_index()

plt.figure(figsize=(7, 5))
sns.barplot(data=survival_by_sex, x='sex', y='survived')
plt.title('Survival Rate by Sex')
plt.xlabel('Sex')
plt.ylabel('Survival Rate')
plt.ylim(0, 1)
plt.show()
#Interpretation: Survival rates differ substantially between male and female passengers in this dataset. 
# The female group has a considerably higher survival rate than the male group. 
# This indicates that sex is an important variable associated with survival, although this chart alone does not explain the role of passenger class or other factors.

# -----------------------------------------
# Chart 2: Survival rate by pclass
# -----------------------------------------
survival_by_pclass = df.groupby('pclass')['survived'].mean().reset_index()

plt.figure(figsize=(7, 5))
sns.barplot(data=survival_by_pclass, x='pclass', y='survived')
plt.title('Survival Rate by pclass')
plt.xlabel('Pclass')
plt.ylabel('Survival Rate')
plt.ylim(0, 1)
plt.show()

#Interpretation:Survival rates differ substantially between the passenger class which is class1,class2 and closs3 in this dataset. first-class passengers generally having higher survival rates than third-class passengers ,
#This indicates that passger class alos is an important variable associated with survival.

# -----------------------------------------
# Chart 3 : Survival by pclass and sex
# -----------------------------------------
plt.figure(figsize=(7, 5))
sns.barplot(data=df, x='pclass', y='survived',hue='sex')
plt.title('Survival Rate by pclass and sex')
plt.xlabel('Pclass')
plt.ylabel('Survival Rate')
plt.ylim(0, 1)
plt.show()

#Interpretation: Survival varies across passenger classes for both male and female,
# female with first-class passengers generally having higher survival rates than male with all the passengers class. The difference between male and female survival rates remains visible within each passenger class. 
# This suggests that survival was associated with both male and female represented by passenger class.

# -----------------------------------------
# Chart 4: Survival by age ,fare,sex
# -----------------------------------------
plt.figure(figsize=(9, 6))

sns.scatterplot(
    data=df,
    x='age',
    y='fare',
    hue='survived',
    style='sex',
    alpha=0.7
)

plt.title('Age, Fare, Sex and Survival')
plt.xlabel('Age')
plt.ylabel('Fare')
plt.show()

#Interpretation: The scatter plot combines age and fare while distinguishing passengers by survival and sex. Higher fares are associated with passengers in higher passenger classes having the more survival chances. The plot therefore provides a multivariate view showing that survival was not simply an age-related outcome but occurred within a broader combination of sex and fare

#standardize age and fare using the z-score formula z = (x − mean) / std on the full cleaned DataFrame

#column to standardize age and columns
numeric_colums =["age", "fare"]

#StandardScaler 
scaler = StandardScaler()
#apply z-score formula z=(x−mean)/std or StandardScaleron the full cleaned DataFrame
df_scaled = df.copy()

df_scaled[numeric_colums] = scaler.fit_transform(df_scaled[numeric_colums])

#before/after comparison (e.g., a printed summary of means/stds)

print("Before scaling")
print(df[numeric_colums].describe().round(2).loc[['mean', 'std']])

print("after scaling")
print(df_scaled[numeric_colums].describe().round(2).loc[['mean', 'std']])

#Split the data into train/test sets first, using a stratified split,Use survived as the classification target.
X=df.drop(columns=["survived"])
y= df["survived"]

X_train,X_test,y_train,y_test= train_test_split(X, y,
                                                test_size=0.2,
                                                random_state=42,stratify=y)
print(f"Train: {X_train.shape[0]} rows ({X_train.shape[0]/len(X)*100:.0f}%)")
print(f"Test:  {X_test.shape[0]} rows ({X_test.shape[0]/len(X)*100:.0f}%)")

comparison = pd.DataFrame({
    'Full Dataset': y.value_counts(normalize=True),
    'Train': y_train.value_counts(normalize=True),
    'Test': y_test.value_counts(normalize=True)
})

print((comparison * 100).round(2))
df.shape
#encode categorical columns (sex, embarked) with label or one-hot encoding, and scale numeric features with StandardScaler,
#Every preprocessing step (imputer, encoder, scaler) must be fit only on the training split, then applied in transform-only mode to the test split.

# split the columns in to categorical columns and numeric_features
categorical_columns = ["sex","embarked"]
numeric_features= ["age","pclass","sibsp","parch","fare"]

feature_col = numeric_features + categorical_columns
target_col = "survived"

df= df[feature_col + [target_col]].copy()
df = df.dropna(subset=[target_col])
df[target_col]= df[target_col].astype(int)

X = df[feature_col]
y = df[target_col]

#train test split before fit and transform
X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=42)
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

numeric_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

pre_preprocessor = ColumnTransformer([
    ('categorical', categorical_pipeline, categorical_columns),
    ('numeric', numeric_pipeline, numeric_features)
])

model_pipeline = Pipeline([
    ('preprocessor', pre_preprocessor),
    ('classifier', LogisticRegression(max_iter=1000))
])

#fit the data only on the training split
model_pipeline.fit(X_train, y_train)

#apply in transform-only mode to the test split
y_pred= model_pipeline.predict(X_test)
y_proba= model_pipeline.predict_proba(X_test)[:,1]

#Train three classifiers on the same train/test split: 
#Logistic Regression, Decision Tree, and Random Forest.
Logistic_pipeline = Pipeline([
    ('preprocessor', pre_preprocessor),
    ('classifier', LogisticRegression(max_iter=1000,random_state=42))
])

Random_forest_pipeline= Pipeline([
    ('preprocessor', pre_preprocessor),
    ('classifier', RandomForestClassifier(max_depth=4,n_estimators =200,random_state=42))
])

DecisionTreeClassifier_pipeline= Pipeline([
    ('preprocessor', pre_preprocessor),
    ('classifier', DecisionTreeClassifier(max_depth=4, min_samples_leaf=10,
                                          random_state=42))
])

#Train three classifiers on the same train/test split
Logistic_pipeline.fit(X_train,y_train)
Random_forest_pipeline.fit(X_train,y_train)
DecisionTreeClassifier_pipeline.fit(X_train,y_train)



# Extract fitted components
fitted_preprocessor = DecisionTreeClassifier_pipeline.named_steps['preprocessor']
decision_tree = DecisionTreeClassifier_pipeline.named_steps['classifier']

# Get feature names after preprocessing
feature_names = fitted_preprocessor.get_feature_names_out()

fix, ax = plt.subplots(figsize=(20, 10))

plot_tree(decision_tree,
          feature_names = feature_names,
          class_names = ['Dead', 'Survived'],
          filled = True, # Coloring the nodes by majortiy calss)
          rounded = True,
          fontsize = 9,
          ax = ax
)

ax.set_title("Decision Tree -  Titanic Survival Prediciton",
             fontsize = 16, fontweight = 'bold')
plt.tight_layout()
plt.show()

#confusion matrix, accuracy, precision, recall, F1 score, ROC curve with AUC

# Logistic Regression prediction
y_pred_LR= Logistic_pipeline.predict(X_test)
y_proba_LR= Logistic_pipeline.predict_proba(X_test)[:,1]

# Decision Tree predictions
y_pred_DT= DecisionTreeClassifier_pipeline.predict(X_test)
y_proba_DT= DecisionTreeClassifier_pipeline.predict_proba(X_test)[:,1]

# Random Forest predictions
y_pred_RF= Random_forest_pipeline.predict(X_test)
y_proba_RF= Random_forest_pipeline.predict_proba(X_test)[:,1]

#Logistic Regression
CM_LR = confusion_matrix(y_test,y_pred_LR)
Accuracy_LR =accuracy_score(y_test,y_pred_LR)
Precision_LR =precision_score(y_test,y_pred_LR)
Recall_LR =recall_score(y_test,y_pred_LR)
F1_LR =f1_score(y_test,y_pred_LR)
fpr_LR, tpr_LR, thresholds_LR = roc_curve(y_test,y_proba_LR)
auc_LR = roc_auc_score(y_test,y_proba_LR)

print("Logistic Regression")
print(f"Accuracy : {Accuracy_LR:.2f}")
print(f"Precision: {Precision_LR:.2f}")
print(f"Recall   : {Recall_LR:.2f}")
print(f"F1 Score : {F1_LR:.2f}")
print(f"AUC      : {auc_LR:.2f}")
print("\nConfusion Matrix:")
print(CM_LR)
# Decision Tree

CM_DT = confusion_matrix(y_test,y_pred_DT)
Accuracy_DT =accuracy_score(y_test,y_pred_DT)
Precision_DT =precision_score(y_test,y_pred_DT)
Recall_DT =recall_score(y_test,y_pred_DT)
F1_DT =f1_score(y_test,y_pred)
fpr_DT, tpr_DT, thresholds_DT = roc_curve(y_test,y_proba_DT)
auc_DT = roc_auc_score(y_test,y_proba_DT)

print("Decision Tree")
print(f"Accuracy : {Accuracy_DT:.2f}")
print(f"Precision: {Precision_DT:.2f}")
print(f"Recall   : {Recall_DT:.2f}")
print(f"F1 Score : {F1_DT:.2f}")
print(f"AUC      : {auc_DT:.2f}")
print("\nConfusion Matrix:")
print(CM_DT)

#Random Forest
CM_RF = confusion_matrix(y_test,y_pred_RF)
Accuracy_RF =accuracy_score(y_test,y_pred_RF)
Precision_RF =precision_score(y_test,y_pred_RF)
Recall_RF =recall_score(y_test,y_pred_RF)
F1_RF =f1_score(y_test,y_pred)
fpr_RF, tpr_RF, thresholds_RF = roc_curve(y_test,y_proba_RF)
auc_RF = roc_auc_score(y_test,y_proba_RF)

print("Random Forest")
print(f"Accuracy : {Accuracy_RF:.2f}")
print(f"Precision: {Precision_RF:.2f}")
print(f"Recall   : {Recall_RF:.2f}")
print(f"F1 Score : {F1_RF:.2f}")
print(f"AUC      : {auc_RF:.2f}")
print("\nConfusion Matrix:")
print(CM_RF)

comparison = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'Decision Tree',
        'Random Forest'
    ],

    'Confusion Matrix': [
        CM_LR.tolist(),
        CM_DT.tolist(),
        CM_RF.tolist()
    ],

    'Accuracy': [
        Accuracy_LR,
        Accuracy_DT,
        Accuracy_RF
    ],

    'Precision': [
        Precision_LR,
        Precision_DT,
        Precision_RF
    ],

    'Recall': [
        Recall_LR,
        Recall_DT,
        Recall_RF
    ],

    'F1 Score': [
        F1_LR,
        F1_DT,
        F1_RF
    ],

    'AUC': [
        auc_LR,
        auc_DT,
        auc_RF
    ]
})

# Round the numerical columns
comparison[
    ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
] = comparison[
    ['Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC']
].round(3)

display(comparison)


#Report survived/not-survived class balance,
# Class counts
class_counts = y.value_counts().sort_index()

# Class percentages
class_percent = y.value_counts(normalize=True).sort_index() * 100

balance = pd.DataFrame({
    'Class': ['Not Survived', 'Survived'],
    'Count': [
        class_counts.get(0, 0),
        class_counts.get(1, 0)
    ],
    'Percentage': [
        class_percent.get(0, 0),
        class_percent.get(1, 0)
    ]
})

balance['Percentage'] = balance['Percentage'].round(2)

print(balance)

#baseline/no handling
baseline_model = Pipeline([
    ('preprocessor', pre_preprocessor),
    ('classifier', LogisticRegression(max_iter=1000,random_state=42))
])

baseline_model.fit(X_train, y_train)
y_pred_baseline = baseline_model.predict(X_test)

#class_weight='balanced' model
balanced_model = Pipeline([
    ('preprocessor', pre_preprocessor),
    ('classifier', LogisticRegression(max_iter=1000,random_state=42,class_weight='balanced'))
])
balanced_model.fit(X_train, y_train)
y_pred_balanced = balanced_model.predict(X_test)

#SMOTE oversampling
smote_model = ImbPipeline([
    ('preprocessor', pre_preprocessor),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(max_iter=1000,
                                      random_state=42))
])
smote_model.fit(X_train, y_train)
y_pred_smote = smote_model.predict(X_test)

comparison = pd.DataFrame({
    'variants': [
        'baseline',
        'class_weight=balanced',
        'SMOTE'
    ],

    'Precision': [
        precision_score(y_test,y_pred_baseline),
        precision_score(y_test,y_pred_balanced),
        precision_score(y_test,y_pred_smote)
    ],

    'Recall': [
        recall_score(y_test,y_pred_baseline),
        recall_score(y_test,y_pred_balanced),
        recall_score(y_test,y_pred_smote)
    ],

    'F1 Score': [
        f1_score(y_test,y_pred_baseline),
        f1_score(y_test,y_pred_balanced),
        f1_score(y_test,y_pred_smote)
    ],

})

# Round the numerical columns
comparison[
    [ 'Precision', 'Recall', 'F1 Score']
] = comparison[
    ['Precision', 'Recall', 'F1 Score']
].round(3)

display(comparison)

import joblib

# Save complete pipeline
joblib.dump(
    Random_forest_pipeline,
    'best_titanic_classifier.joblib'
)

print("Best model: Decision Tree")
print("Complete pipeline saved successfully.")

'''Interpretation
Decision Tree has the highest accuracy (0.815).
It also has the highest precision (0.781), meaning its positive survival predictions are the most precise among the three.
It has the highest ROC-AUC (0.865), indicating the strongest discrimination between survived and not-survived cases in this test set.
Logistic Regression has the highest recall (0.783), so it identifies more of the actual survivors.
All three models have the same rounded F1 = 0.73; therefore F1 alone does not distinguish them at the precision shown '''

#Hyperparameter tuning: 

# Random Forest with OOB scoring enabled
rf_oob= RandomForestClassifier(
    oob_score=True,
    random_state=42,
    n_jobs=1
)

# Keep preprocessing + Random Forest together
rf_grid_pipeline=Pipeline([
    ("preprocessor", pre_preprocessor),
    ("classifier", rf_oob)
])

## --- Hyperparameter grid ---
param_grid={
    "n_estimators":[100,200,500],
    "max_depth": [None,5,10,20],
    "max_features":["sqrt","log2","0.5"]
}
#GridSearchCV
grid_search = GridSearchCV(
    estimator =rf_grid_pipeline,
    param_grid =param_grid,
    cv = 5,
    scoring = 'f1',
    n_jobs=-1,
    refit=True
 )

#Regression side-task:
#using the same dataset,predict fare from the other available features with a multivariate linear regression

categorical_columns_reg = ["sex","embarked"]
numerical_features_reg= ["survived","age","pclass","sibsp","parch"]

X_rg=df.drop(columns=["fare"])
y_rg= df['fare']

X_train_rg,X_test_rg,y_train_rg,y_test_rg = train_test_split(X_rg,y_rg,
                                          test_size=0.2,random_state=42)
print("Training shape:", X_train_rg.shape)
print("Testing shape:", X_test_rg.shape)
print(f"Train: {X_train_rg.shape[0]} rows ({X_train_rg.shape[0]/len(X)*100:.0f}%)")
print(f"Test:  {X_test_rg.shape[0]} rows ({X_test_rg.shape[0]/len(X)*100:.0f}%)")                                      

#fit model on train data
linear_model.fit(X_train_rg, y_train_rg)
# do prediction on test data

y_test_pred_rg = linear_model.predict(X_test_rg)

#calculare MAE,RMSE and R2_score values
MAE = mean_absolute_error(y_test_rg, y_test_pred_rg)
RMSE = np.sqrt(mean_squared_error(y_test_rg, y_test_pred_rg))
R2  = r2_score(y_test_rg, y_test_pred_rg)

# Number of test observations
n= len(y_test_rg)

# Number of predictors after one-hot encoding
X_test_transformed = linear_model.named_steps[
    'preprocessor'
].transform(X_test_rg)
p=X_test_transformed.shape[1]

# Adjusted R²
Adjusted_R2 = 1 - (
    (1 - R2) * (n - 1) / (n - p - 1)
)

print(f"MAE  : {MAE:.3f}")
print(f"RMSE : {RMSE:.3f}")
print(f"R²   : {R2:.3f}")
print(f"Adjusted_R2 : {Adjusted_R2:.3f}")

#produce a residual plot, stating in writing whether it shows heteroscedasticity
#Residual=Actual Fare−Predicted Fare
residual = y_test_rg-y_test_pred_rg

plt.figure(figsize=(8, 6))

sns.scatterplot(
    x=y_test_pred_rg,
    y=residual,
    alpha=0.6
)
plt.axhline(
    y=0,
    linestyle='--'
)
plt.xlabel("Predicted Fare")
plt.ylabel("Residual")
plt.title("Residual plot-multivariate linear regression")
plt.tight_layout()
plt.show()

''''
Residual plot shows heteroscedasticity (a non-random spread of residuals)since variance is relatively constant, 
which is consistent with homoscedasticity by analysing the visual evidence of Residual plot
'''

'''
Write a model comparison table that presents the three classifiers' metrics (accuracy, precision, recall, F1, AUC) side by side,
and the regression model's metrics (MAE, RMSE, R², Adjusted R²) side by side as their own separate columns.
'''

model_comparison = pd.DataFrame({
    'Model': [
        'Logistic Regression',
        'Decision Tree',
        'Random Forest',
        'Linear Regression'
    ],

    'Clasification-Accuracy': [
        Accuracy_LR,Accuracy_DT,Accuracy_RF,None
        ],

    'Clasification-Precision': [
        Precision_LR,Precision_DT,Precision_RF,None
        ],
    
    'Clasification-Recall': [
        Recall_LR,Recall_DT,Recall_RF,None
        ],
    'Clasification-F1 Score': [
        F1_LR,F1_DT,F1_RF,None
        ],
    'Clasification--AUC': [
        auc_LR,auc_DT,auc_RF,None
        ],
    'Regression-MAE' :[
        None,None,None,MAE
        ],
    'Regression-RMSE' :[
        None,None,None,RMSE
        ],
    'Regression-R2' :[
        None,None,None,R2
        ],
    'Regression-Adjusted_R2' :[
        None,None,None,Adjusted_R2
        ],
})


# Round the numerical columns
model_comparison.iloc[:, 1:] = model_comparison.iloc[:, 1:].round(3)

display(model_comparison)







