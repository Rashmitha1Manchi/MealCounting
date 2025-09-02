import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE

df=pd.read_csv("meal_data.csv")
df.isnull().sum()

#Convert timestamp to datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

#Feature engineering
df['hour'] = df['timestamp'].dt.hour
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['day'] = df['timestamp'].dt.day
df = df.drop(columns=['timestamp'])

#Encode categorical variables
le_student = LabelEncoder()
df['student_id'] = le_student.fit_transform(df['student_id'])

le_status = LabelEncoder()
df['status'] = le_status.fit_transform(df['status'])

le_meal = LabelEncoder()
df['meal_type_encoded'] = le_meal.fit_transform(df['meal_type'])

#Define features and target
X = df[['student_id', 'hour', 'dayofweek', 'day']]
y = df['meal_type_encoded']

#Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#  SMOTE to balance the training set
smote = SMOTE(random_state=42)
X_train_res1, y_train_res1 = smote.fit_resample(X_train, y_train)

print("Before SMOTE:", y_train.value_counts().to_dict())
print("After SMOTE:", pd.Series(y_train_res1).value_counts().to_dict())

models1 = {
    # 'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    # 'KNN': KNeighborsClassifier(),
    # 'SVM': SVC(probability=True),
    # 'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='mlogloss')
}

scale_pos_weight = (y_train.value_counts()[0] / y_train.value_counts()[1])

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),    
    'KNN': KNeighborsClassifier(),
    'SVM': SVC(probability=True),
    'XGBClassifier' : XGBClassifier(
        use_label_encoder=False,
        eval_metric='mlogloss',
        random_state=42,
        scale_pos_weight=scale_pos_weight
    )
}

results = []

#With SMOTE
for name, model in models1.items():
    model.fit(X_train_res1, y_train_res1)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    results.append((name, acc, f1))
    print(f"\n {name}")
    print("Accuracy:", acc)
    print("F1 Score (macro):", f1)
    print(classification_report(y_test, y_pred, target_names=le_meal.classes_))


#Without SMOTE
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    results.append((name, acc, f1))
    print(f"\n {name}")
    print("Accuracy:", acc)
    print("F1 Score (macro):", f1)
    print(classification_report(y_test, y_pred, target_names=le_meal.classes_))


results_df = pd.DataFrame(results, columns=['Model', 'Accuracy', 'F1 Score']).sort_values(by='F1 Score', ascending=False)

plt.figure(figsize=(10,5))
sns.barplot(x='F1 Score', y='Model', data=results_df, palette='coolwarm')
plt.title('Model Comparison (F1 Score)')
plt.xlim(0, 1)
plt.show()