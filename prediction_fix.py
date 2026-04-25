import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 55)
print("  PELATIHAN MODEL - 15 FITUR TERPILIH")
print("=" * 55)

df = pd.read_csv(r'C:\Users\aryod\Downloads\data_student.csv', sep=';')
print(f"\n[INFO] Data dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

df_model = df[df['Status'].isin(['Dropout', 'Graduate'])].copy()
df_model['Status'] = df_model['Status'].map({'Graduate': 0, 'Dropout': 1})
print(f"[INFO] Data setelah filter: {df_model.shape[0]} baris")
print(f"       Graduate : {(df_model['Status']==0).sum()}")
print(f"       Dropout  : {(df_model['Status']==1).sum()}")

SELECTED_FEATURES = [
    'Curricular_units_2nd_sem_approved',
    'Curricular_units_1st_sem_approved',
    'Curricular_units_2nd_sem_grade',
    'Curricular_units_1st_sem_grade',
    'Tuition_fees_up_to_date',
    'Admission_grade',
    'Age_at_enrollment',
    'Course',
    'Previous_qualification_grade',
    'Curricular_units_2nd_sem_evaluations',
    'Curricular_units_1st_sem_evaluations',
    'Fathers_occupation',
    'Scholarship_holder',
    'Gender',
    'Debtor',
]

X = df_model[SELECTED_FEATURES]
y = df_model['Status']
print(f"\n[INFO] Fitur yang digunakan ({len(SELECTED_FEATURES)}):")
for i, f in enumerate(SELECTED_FEATURES, 1):
    print(f"       {i:2d}. {f}")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"\n[INFO] Train: {len(X_train)} | Test: {len(X_test)}")

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

print("\n[INFO] Melatih Random Forest Classifier...")
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train_scaled, y_train)
print("[INFO] Pelatihan selesai!")

y_pred   = rf_model.predict(X_test_scaled)
accuracy = accuracy_score(y_test, y_pred)

print("\n" + "=" * 55)
print("  HASIL EVALUASI MODEL")
print("=" * 55)
print(f"\nAkurasi: {accuracy * 100:.2f}%\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Graduate', 'Dropout']))

importances   = rf_model.feature_importances_
indices       = np.argsort(importances)[::-1]
feature_names = X.columns.tolist()

print("Top 15 Feature Importance:")
for i in range(15):
    print(f"  {i+1:2d}. {feature_names[indices[i]]:<45} {importances[indices[i]]:.4f}")

plt.figure(figsize=(12, 5))
plt.title("Top 15 Fitur Paling Berpengaruh", fontsize=13)
plt.bar(range(15), importances[indices][:15], color='teal')
plt.xticks(range(15), [feature_names[i] for i in indices[:15]], rotation=45, ha='right')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150)
plt.show()

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Graduate', 'Dropout'],
            yticklabels=['Graduate', 'Dropout'])
plt.title('Confusion Matrix')
plt.ylabel('Aktual')
plt.xlabel('Prediksi')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=150)
plt.show()

joblib.dump(rf_model, 'model_rf.pkl')
joblib.dump(scaler,   'scaler.pkl')
print("\n[INFO] Model disimpan : model_rf.pkl")
print("[INFO] Scaler disimpan: scaler.pkl")
print("\n[SELESAI] Jalankan: streamlit run app.py")