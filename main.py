import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, recall_score, precision_score, f1_score

def analyze_telco_churn(filepath):
    # 1. Chargement et Nettoyage
    global df
    df = pd.read_csv(filepath)
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["MonthlyCharges"] * df["tenure"])
    df = df.dropna()

    lignes_manquantes = df[df.isnull().any(axis=1)]
    print("\nLignes avec des valeurs manquantes :")
    print(lignes_manquantes)
    
    # --- 1. QUEL EST LE TAUX DE CHURN GLOBAL ? ---
    print(" TAUX DE CHURN GLOBAL")
    print("-" * 40)
    churn_counts = df["Churn"].value_counts()
    churn_pct = df["Churn"].value_counts(normalize=True) * 100
    print(f"Clients Restants (No) : {churn_counts['No']} ({churn_pct['No']:.2f}%)")
    print(f"Clients Partis (Yes)  : {churn_counts['Yes']} ({churn_pct['Yes']:.2f}%)")
    
    plt.figure(figsize=(8, 6))
    colors = ['#2ecc71', '#e74c3c']
    plt.pie(churn_counts, labels=['Non (Restants)', 'Oui (Partis)'], autopct='%1.1f%%', 
            colors=colors, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
    plt.title("Repartition Globale du Churn", fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.show()

    # --- 2. LES NOUVEAUX CLIENTS QUITTENT-ILS PLUS ? ---
    print("LES NOUVEAUX CLIENTS QUITTENT-ILS PLUS ?")
    print("-" * 40)
    tenure_bins = [0, 12, 24, 36, 48, 60, 72]
    tenure_labels = ['0-12', '13-24', '25-36', '37-48', '49-60', '61-72']
    df['tenure_group'] = pd.cut(df['tenure'], bins=tenure_bins, labels=tenure_labels, include_lowest=True, right=True)
    tenure_churn = df.groupby('tenure_group')['Churn'].apply(lambda x: (x == 'Yes').mean() * 100)

    for label in tenure_labels:
        print(f"Churn {label} mois : {tenure_churn.loc[label]:.2f}%")

    plt.figure(figsize=(12, 6))
    sns.barplot(x=tenure_churn.index, y=tenure_churn.values, palette='viridis')
    plt.title("Taux de churn par groupe d'ancienneté", fontsize=14, weight='bold', pad=20)
    plt.xlabel("Groupe de tenure (mois)", fontsize=12)
    plt.ylabel("Taux de churn (%)", fontsize=12)
    plt.ylim(0, tenure_churn.max() * 1.15)
    for i, value in enumerate(tenure_churn.values):
        plt.text(i, value + 1, f"{value:.1f}%", ha='center', va='bottom', fontsize=10, weight='bold')
    plt.tight_layout()
    plt.show()

    # --- 3. LE TYPE DE CONTRAT INFLUENCE-T-IL LE CHURN ? ---
    print("LE TYPE DE CONTRAT INFLUENCE-T-IL LE CHURN ?")
    print("-" * 40)
    contract_churn = pd.crosstab(df["Contract"], df["Churn"], normalize='index') * 100
    print(contract_churn.round(2))
    
    plt.figure(figsize=(10, 6))
    sns.countplot(data=df, x="Contract", hue="Churn", palette=['#2ecc71', '#e74c3c'],
                  edgecolor='black', linewidth=1.2)
    plt.title("Churn par Type de Contrat", fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.show()

    # --- 4. LE PRIX INFLUENCE-T-IL LE CHURN ? ---
    print(" LE PRIX INFLUENCE-T-IL LE CHURN ?")
    print("-" * 40)
    price_no = df[df["Churn"] == "No"]["MonthlyCharges"].mean()
    price_yes = df[df["Churn"] == "Yes"]["MonthlyCharges"].mean()
    print(f"Prix moyen (Clients restants) : ${price_no:.2f}")
    print(f"Prix moyen (Clients partis) : ${price_yes:.2f}")
    
    plt.figure(figsize=(12, 5))
    sns.kdeplot(data=df, x="MonthlyCharges", hue="Churn", fill=True, 
                palette=['#2ecc71', '#e74c3c'], thresh=0)
    plt.title("Influence du Prix Mensuel sur le Churn", fontsize=14, weight='bold', pad=20)
    plt.tight_layout()
    plt.show()

    # --- 5. LE SUPPORT TECHNIQUE INFLUENCE-T-IL LE CHURN ? ---
    print("LE SUPPORT TECHNIQUE INFLUENCE-T-IL LE CHURN ?")
    print("-" * 40)
    
    
    tech_support_churn = pd.crosstab(df["TechSupport"], df["Churn"], normalize='index') * 100
    print("Support Technique :")
    print(tech_support_churn.round(2))
    
    plt.figure(figsize=(12, 5))
    sns.countplot(data=df, x="TechSupport", hue="Churn", palette=['#2ecc71', '#e74c3c'])
    plt.title("Support technique", fontsize=14, weight='bold', pad=20)
    plt.xlabel("sup tech", fontsize=11)
    plt.ylabel("nombre client", fontsize=11)
    plt.legend(title="Churn", labels=["Non", "Oui"])
    plt.tight_layout()
    plt.show()

    # --- 6. QUELS PROFILS DE CLIENTS QUITTENT LE PLUS ? ---
    print("QUELS PROFILS DE CLIENTS QUITTENT LE PLUS ?")
    print("-" * 40)
    churn_by_senior = df.groupby("SeniorCitizen")["Churn"].apply(lambda x: (x == "Yes").mean() * 100)
    churn_by_partner = df.groupby("Partner")["Churn"].apply(lambda x: (x == "Yes").mean() * 100)
    churn_by_dependents = df.groupby("Dependents")["Churn"].apply(lambda x: (x == "Yes").mean() * 100)
    churn_by_payment = df.groupby("PaymentMethod")["Churn"].apply(lambda x: (x == "Yes").mean() * 100)

    print(f"Seniors (SeniorCitizen=1) : {churn_by_senior.loc[1]:.1f} % de churn vs {churn_by_senior.loc[0]:.1f} % pour les non-seniors")
    print(f"Clients sans partenaire : {churn_by_partner.loc['No']:.1f} % de churn vs {churn_by_partner.loc['Yes']:.1f} % pour les clients avec partenaire")
    print(f"Clients sans enfants à charge : {churn_by_dependents.loc['No']:.1f} % de churn vs {churn_by_dependents.loc['Yes']:.1f}")
    print(f"Paiement par chèque électronique : {churn_by_payment.loc['Electronic check']:.1f} % de churn (le plus élevé)")

    profile_df = pd.DataFrame({
        'Segment': [
            'Seniors', 'Non-seniors',
            'Sans partenaire', 'Avec partenaire',
            'Sans enfants', 'Avec enfants',
            'Electronic check', 'Autres modes de paiement'
        ],
        'Churn': [
            churn_by_senior.loc[1], churn_by_senior.loc[0],
            churn_by_partner.loc['No'], churn_by_partner.loc['Yes'],
            churn_by_dependents.loc['No'], churn_by_dependents.loc['Yes'],
            churn_by_payment.loc['Electronic check'], churn_by_payment.drop('Electronic check').mean()
        ]
    })

    plt.figure(figsize=(12, 6))
    sns.barplot(data=profile_df, x='Segment', y='Churn', palette=['#c0392b', '#27ae60', '#d35400', '#2980b9', '#8e44ad', '#16a085', '#c0392b', '#95a5a6'])
    plt.title("Profils de clients les plus exposés au churn", fontsize=14, weight='bold', pad=20)
    plt.ylabel("Taux de churn (%)", fontsize=12)
    plt.xlabel("")
    plt.xticks(rotation=30, ha='right')
    for p in plt.gca().patches:
        plt.gca().annotate(f"{p.get_height():.1f}%", (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center', xytext=(0, 9), textcoords='offset points', fontsize=10)
    plt.tight_layout()
    plt.show()
    
    print("\n" + "="*60)
    print("FIN DE L'ANALYSE")
    print("="*60 + "\n")

    return df


df_clean = analyze_telco_churn("Telco_Customer_Churn.csv")

print("CONSTRUCTION DU GRAPHIQUE DE SYNTHeSE VERTICAL...")
df['tenure_group'] = np.where(df['tenure'] <= 10, 'Nouveau (<= 10m)', 'Ancien (> 10m)')


target_vars = {
    'tenure_group': 'Anciennete',
    'Contract': 'Contrat',
    'PaymentMethod': 'Mode Paiement',
    'InternetService': 'Internet',
    'OnlineSecurity': 'Securite Online',
    'TechSupport': 'Support Tech'
}

results = []
for col, label in target_vars.items():
    summary = df.groupby(col)['Churn'].apply(lambda x: (x == 'Yes').mean() * 100).reset_index()
    summary.columns = ['Valeur', '% Churn']
    summary['Variable'] = label
    results.append(summary)

df_impact = pd.concat(results).sort_values(by='% Churn', ascending=False)


plt.figure(figsize=(14, 8))
sns.set_style("whitegrid")

ax = sns.barplot(data=df_impact, x='Valeur', y='% Churn', hue='Variable', dodge=False)


for p in ax.patches:
    if p.get_height() > 0:
        ax.annotate(f'{p.get_height():.1f}%', 
                    (p.get_x() + p.get_width() / 2., p.get_height()), 
                    ha='center', va='center', xytext=(0, 9), 
                    textcoords='offset points', fontsize=10, weight='bold')

plt.title("Synthese : Facteurs d'Influence Majeurs sur le Churn", fontsize=16, weight='bold', pad=20)
plt.ylabel("Taux de Churn (%)", fontsize=12, weight='bold')
plt.xlabel("Segments et Services", fontsize=12, weight='bold')
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 60)
plt.legend(title="Variable d'origine", bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
print("ANALYSE TERMINeE.")


print("\n" + "="*40)
print("PARTIE 3 : ANALYSE DE CORReLATION")
print("="*40)
df_ml = df.copy()
df_ml['Churn'] = df_ml['Churn'].map({'Yes': 1, 'No': 0})
if 'customerID' in df_ml.columns:
    df_ml = df_ml.drop('customerID', axis=1)
df_encoded = pd.get_dummies(df_ml, drop_first=True)

# Correlation avec la cible (Churn)
plt.figure(figsize=(10, 8))
correlation = df_encoded.corr()['Churn'].sort_values(ascending=False)
correlation.drop('Churn').plot(kind='barh', color='skyblue')
plt.title("Correlation des variables avec le Churn")
plt.show()

print("Top 3 facteurs positifs (favorisent le churn) :\n", correlation.head(4)[1:])
print("\nTop 3 facteurs negatifs (favorisent la retention) :\n", correlation.tail(3))

print("\n" + "="*40)
print("PARTIE 4 : MODeLE PReDICTIF")
print("="*40)

# 1. Selection des variables
X = df_encoded.drop('Churn', axis=1)
y = df_encoded['Churn']

# 2. Train/Test Split (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 3. Scaling (Indispensable pour la Regression Logistique)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. Entraînement du modele
model = LogisticRegression(max_iter=1000)
model.fit(X_train_scaled, y_train)

# 5. Predictions
y_pred = model.predict(X_test_scaled)

# 6. evaluation
print("\n--- MATRICE DE CONFUSION ---")
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel('Prediction')
plt.ylabel('Realite')
plt.show()

print("\n--- RAPPORT DE PERFORMANCE ---")
print(classification_report(y_test, y_pred))

# ==========================================
# PARTIE 5 — INTERPReTATION MeTIER
# ==========================================
print("\n" + "="*40)
print("PARTIE 5 : INTERPReTATION DES COEFFICIENTS")
print("="*40)

# Analyse des coefficients pour comprendre l'influence reelle
weights = pd.Series(model.coef_[0], index=X.columns).sort_values(ascending=False)

print("Variables les plus impactantes selon le modele :")
print(weights.head(5)) # Facteurs de risque
print(weights.tail(5)) # Facteurs de protection

# Visualisation des coefficients
plt.figure(figsize=(10, 10))
weights.plot(kind='barh')
plt.title("Poids des variables dans la decision du modele")
plt.tight_layout()
plt.show()


# --- CALCUL DES MeTRIQUES ---
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n" + "="*40)
print("eVALUATION DeTAILLeE DU MODeLE")
print("="*40)
print(f"Precision (Accuracy)  : {accuracy:.2f} ")
print(f"Precision (Precision) : {precision:.2f}  ")
print(f"Rappel (Recall)       : {recall:.2f}  ")
print(f"Score F1              : {f1:.2f} ")
print("-" * 40)

# Visualisation des metriques sous forme de graphique pour ton rapport
metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
metrics_values = [accuracy, precision, recall, f1]

plt.figure(figsize=(10, 6))
ax = sns.barplot(x=metrics_names, y=metrics_values, palette='viridis')
plt.ylim(0, 1)
plt.title("Performances du Modele de Regression Logistique", fontsize=14, weight='bold')

for p in ax.patches:
    ax.annotate(f'{p.get_height():.2f}', (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 9), textcoords='offset points', weight='bold')

plt.show()

# Sortie du dataset with predictions
df_test_results = X_test.copy()
df_test_results['Churn_Actual'] = y_test
df_test_results['Churn_Predicted'] = y_pred
df_test_results.to_csv("test_results_with_predictions.csv", index=False)

# Résumé final
n_clients_test = len(df_test_results)
actual_churn_pct = df_test_results['Churn_Actual'].mean() * 100
predicted_churn_pct = df_test_results['Churn_Predicted'].mean() * 100
print("\n" + "="*40)
print("RÉSUMÉ FINAL DU TEST")
print("="*40)
print(f"Nombre de clients en test      : {n_clients_test}")
print(f"Taux de churn réel (Yes)      : {actual_churn_pct:.2f}%")
print(f"Taux de churn prédit (Yes)    : {predicted_churn_pct:.2f}%")
print("="*40)