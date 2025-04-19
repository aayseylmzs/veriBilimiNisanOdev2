import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('abg_dataset_300.csv')

print("Veri seti boyutu:", df.shape)
print("Sütunlar:", df.columns.tolist())
print(df.head())

print("\nTanı (Diagnosis) dağılımı:")
print(df['Diagnosis'].value_counts()) #Bu dağılım, modelimizin sınıfları dengeli öğrenip öğrenemeyeceğini anlamak için önemlidir

print("Eksik değer sayısı (her sütun için):")
print(df.isnull().sum())

#eksik veri doldurma
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
df[categorical_cols] = df[categorical_cols].fillna('Unknown')

print("\nEksik veri kaldı mı?:", df.isnull().values.any())

le_gender = LabelEncoder()
df['Gender'] = le_gender.fit_transform(df['Gender'])

le_diagnosis = LabelEncoder()
df['Diagnosis_encoded'] = le_diagnosis.fit_transform(df['Diagnosis'])

print("\nTanı Kodları (Label Encoding):")
for i, label in enumerate(le_diagnosis.classes_):
    print(f"{i} = {label}")

#veri temizliği
X = df.drop(columns=['Patient ID', 'Sample Time', 'Diagnosis', 'Diagnosis_encoded'])
y = df['Diagnosis_encoded']

#eğitim
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\nEğitim seti boyutu: {X_train.shape}")
print(f"Test seti boyutu: {X_test.shape}")

#modelleme
model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("🔍 Doğruluk (Accuracy):", accuracy_score(y_test, y_pred))
print("🎯 F1 Skoru (macro):", f1_score(y_test, y_pred, average='macro'))
print("\n📋 Sınıflandırma Raporu:")
print(classification_report(y_test, y_pred, target_names=le_diagnosis.classes_))

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8,6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le_diagnosis.classes_, yticklabels=le_diagnosis.classes_, cmap="Blues")
plt.title("📊 Karışıklık Matrisi")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()

#Doğruluk (Accuracy): 0.62
#Bu oran, modelin genel anlamda %62 doğru tahmin yaptığına işaret ediyor. Ancak, F1 skoru düşük olduğundan, bazı sınıflarda çok düşük başarı var. Özellikle "Metabolic Acidosis" ve "Respiratory Alkalosis" gibi az örneklemli sınıflar için model doğru tahmin yapamıyor.
#sınıflar dengesiz olduğundan model çoğunlukla normal olarak tahmin ediyor

# Random Forest modeli
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Özelliklerin önemi
feature_importances = rf_model.feature_importances_

# Özellik adları ve önem düzeylerini göster
feature_data = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importances
})

# Önem sırasına göre sırala
feature_data = feature_data.sort_values(by='Importance', ascending=False)

print(feature_data)

#NİSAN AYI ODEV 1
#KNN MODELİ

knn_model = KNeighborsClassifier(n_neighbors=5)  # k=5 (yakın 5 komşu)
knn_model.fit(X_train, y_train)
#TAHMİN
y_pred_knn = knn_model.predict(X_test)
#DOĞRULUK SKORU

print("🔍 KNN Doğruluk (Accuracy):", accuracy_score(y_test, y_pred_knn))
print("\n📊 Karışıklık Matrisi:")
cm_knn = confusion_matrix(y_test, y_pred_knn)
plt.figure(figsize=(8,6))
sns.heatmap(cm_knn, annot=True, fmt='d', xticklabels=le_diagnosis.classes_, yticklabels=le_diagnosis.classes_, cmap="Blues")
plt.title("KNN - Karışıklık Matrisi")
plt.xlabel("Tahmin Edilen")
plt.ylabel("Gerçek")
plt.show()

# KNN Doğruluk (Accuracy): 0.5166666666666667

# KNN modelinde farklı k değerleri ile optimizasyon yapmak
for k in [3, 5, 7, 9, 11]:
    knn_model = KNeighborsClassifier(n_neighbors=k)
    knn_model.fit(X_train, y_train)
    y_pred_knn = knn_model.predict(X_test)
    print(f"\nK = {k} - Doğruluk:", accuracy_score(y_test, y_pred_knn))


