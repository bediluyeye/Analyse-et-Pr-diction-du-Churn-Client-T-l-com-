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
    tenure_churn = pd.crosstab(df["tenure"] <= 10, df["Churn"], normalize='index') * 100
    print(f"Churn dans les 10 premiers mois : {tenure_churn.loc[True, 'Yes']:.2f}%")
    print(f"Churn apres 10 mois : {tenure_churn.loc[False, 'Yes']:.2f}%")
    
    plt.figure(figsize=(12, 5))
    sns.histplot(data=df, x="tenure", hue="Churn", multiple="stack", bins=40, 
                  palette=['#2ecc71', '#e74c3c'], edgecolor='black')
    plt.title("Churn selon l'Anciennete (Tenure)", fontsize=14, weight='bold', pad=20)
    plt.xlabel("Mois d'anciennete", fontsize=11)
    plt.ylabel("Nombre de clients", fontsize=11)
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
        
    # Visualisation des profils
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    
    sns.countplot(data=df, x="Contract", hue="Churn", ax=axes[0, 0],
                  palette=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=1)
    axes[0, 0].set_title("Churn par Type de Contrat", fontsize=12, weight='bold')
    axes[0, 0].legend(title="Churn", labels=["Non", "Oui"])
    
    sns.countplot(data=df, x="InternetService", hue="Churn", ax=axes[0, 1],
                  palette=['#2ecc71', '#e74c3c'], edgecolor='black', linewidth=1)
    axes[0, 1].set_title("Churn par Type de Connexion", fontsize=12, weight='bold')
    axes[0, 1].legend(title="Churn", labels=["Non", "Oui"])
    
   
    sns.boxplot(data=df, x="Churn", y="MonthlyCharges", hue="Churn", 
                palette=['#2ecc71', '#e74c3c'], legend=False, ax=axes[1, 0])
    axes[1, 0].set_title("Distribution des Prix par Churn", fontsize=12, weight='bold')
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].set_xticklabels(["Non", "Oui"])
    
    sns.boxplot(data=df, x="Churn", y="tenure", hue="Churn", 
                palette=['#2ecc71', '#e74c3c'], legend=False, ax=axes[1, 1])
    axes[1, 1].set_title("Distribution de l'Anciennete par Churn", fontsize=12, weight='bold')
    axes[1, 1].set_xticks([0, 1])
    axes[1, 1].set_xticklabels(["Non", "Oui"])
    
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

#Sortie du dataset with predictions
df_test_results = X_test.copy()
df_test_results['Churn_Actual'] = y_test    
df_test_results['Churn_Predicted'] = y_pred
df_test_results.to_csv("test_results_with_predictions.csv", index=False)