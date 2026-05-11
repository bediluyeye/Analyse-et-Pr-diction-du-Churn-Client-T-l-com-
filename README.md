## 📊 Analyse et Prédiction du Churn - Secteur Télécom
## 📋 Description du Projet
Ce projet vise à résoudre une problématique business majeure pour une entreprise de télécommunication : la perte progressive de clients (churn). L'objectif est de comprendre pourquoi les clients partent, de prédire les départs futurs via le Machine Learning et de fournir des outils de visualisation pour orienter les décisions marketing.
## 🎯 Objectifs

   1. Identifier les facteurs clés influençant le départ des clients.
   2. Construire un modèle prédictif performant (Classification).
   3. Déployer un dashboard interactif pour le suivi des KPI.
   4. Proposer des recommandations stratégiques de fidélisation.

## 🗂️ Dataset
Le projet utilise le dataset Telco Customer Churn disponible sur Kaggle.
Il contient des informations sur :

* Services : Téléphonie, internet, sécurité en ligne, support technique, etc.
* Données clients : Ancienneté (tenure), mode de facturation, type de contrat.
* Facturation : Montant mensuel, montant total.

## 🛠️ Stack Technique

* Analyse & Modélisation : Python (Pandas, Numpy, Matplotlib, Seaborn, Scikit-Learn)
* Machine Learning : Régression Logistique (minimum)
* Visualisation Business : Power BI
* Environnement : Visual Studio Code / Git Bash

## 🚀 Structure du Projet
## 1. Préparation des Données

* Nettoyage des données (gestion des valeurs manquantes, types de colonnes).
* Encodage des variables catégorielles (One-Hot / Label Encoding).

## 2. Analyse Exploratoire (EDA)
Réponses aux questions clés :

* Le profil des clients qui quittent le service (Type de contrat, prix, support).
* Relation entre l'ancienneté et le taux de churn.

## 3. Modélisation Prédictive

* Split des données (Train/Test).
* Entraînement d'un modèle de classification.
* Évaluation des performances via la matrice de confusion, l'Accuracy, la Précision et le F1-Score.

## 4. Export & Business Intelligence

* Export des prédictions vers un fichier churn_cleaned.csv.
* Création d'un dashboard Power BI sur 4 pages :
* Vue Globale (KPI principaux).
* Analyse des Facteurs (Impact des services et contrats).
* Segmentation Clients (Profils à risque).
* Prédictions (Ciblage des clients susceptibles de partir).

## 📈 Résultats & Recommandations


## ⚙️ Installation

   1. Clonez le repository :
   
   git clone https://github.com/bediluyeye/Analyse-et-Pr-diction-du-Churn-Client-T-l-com-.git
   
   2. Installez les dépendances :
   
   pip install -r requirements.txt



