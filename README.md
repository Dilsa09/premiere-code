TP PYTHON — INTELLIGENCE ARTIFICIELLE EN FINANCE
A.	Larhlimi
Chapitres 1, 2 et 3 : Fondements et Applications

📋 INFORMATIONS TP
Cours : Intelligence Artificielle - 
Établissement : ENCG Settat - 4ème année
Professeur : A. Larhlimi
Durée : 3 heures
Modalités : Travail individuel, rendu code Python + rapport PDF

🎯 OBJECTIFS PÉDAGOGIQUES
À l'issue de ce TP, vous serez capable de :
1.	Manipuler les distributions statistiques (moyenne, variance, loi normale) et calculer des métriques Finance (VaR, Sharpe)
2.	Appliquer le théorème de Bayes pour mettre à jour des probabilités de risque (scoring crédit)
3.	Construire et évaluer un modèle supervisé KNN pour classification crédit avec optimisation hyperparamètres
4.	Interpréter des métriques ML (confusion matrix, AUC, ROC) dans un contexte métier Finance


PARTIE 1 — STATISTIQUES ET LOI NORMALE EN FINANCE
Analyse risque portefeuille et calcul VaR
Durée estimée : 45 minutes | Points : 30/100

📝 ÉNONCÉ PARTIE 1
Contexte métier
Vous êtes analyste risques dans une banque d'investissement. Votre mission : analyser les rendements historiques de deux portefeuilles actions pour conseiller un client :
 	Portefeuille A (CONSERVATIVE) : Actions blue-chip européennes (CAC 40, DAX)
 	Portefeuille B (AGRESSIF) : Actions small-cap tech émergentes
Le client dispose de €500,000 à investir et tolère une perte maximale de €50,000 (10% capital) sur un horizon annuel avec 95% de confiance.

Données historiques (rendements mensuels %, 24 mois)
# Portefeuille A (Conservative) rendements_A = np.array([
1.2, 0.8, -0.5, 1.5, 0.9, 1.1, 0.7, 1.3, 1.0, 0.6, 1.4, 0.8,
1.1, 0.9, -0.3, 1.2, 1.0, 1.5, 0.8, 1.3, 0.9, 1.1, 1.2, 1.0
])

# Portefeuille B (Agressif) rendements_B = np.array([
4.5, -2.1, 6.2, -3.5, 5.8, 7.1, -1.8, 4.9, 3.2, -4.2, 8.5, -2.7,
5.1, 6.8, -3.1, 7.3, 4.5, -2.9, 6.7, 5.3, -3.8, 7.9, 4.2, 5.5
])


QUESTIONS (Partie 1)

Question 1.1 — Statistiques descriptives (8 points)

Pour chaque portefeuille, calculez et affichez :
a)	Moyenne des rendements mensuels (%)
b)	Écart-type des rendements mensuels (%)
c)	Médiane des rendements
d)	Rendement annuel : R_{annuel} = (1 + R_{mensuel})^{12} - 1 (formule capitalisation)
e)	Volatilité annuelle : \sigma_{annuel} = \sigma_{mensuel} \times \sqrt{12}
📌 Format attendu :
PORTEFEUILLE A (Conservative)
•	Rendement mensuel moyen : X.XX%
•	Écart-type mensuel : X.XX%
•	Médiane : X.XX%
•	Rendement annualisé : X.XX%
•	Volatilité annualisée : X.XX%
Question 1.2 — Visualisation distributions (6 points)

Créez une figure avec 2 subplots :
a)	Subplot 1 : Histogrammes superposés des rendements A (vert) et B (rouge) avec légendes
b)	Subplot 2 : Boxplots comparatifs A vs B (identifier outliers)
📌 Éléments requis : Titres, labels axes, grille, légendes


Question 1.3 — Value at Risk (VaR 95%) (10 points)

Pour chaque portefeuille, calculez la VaR 95% mensuelle et annuelle :
Formule VaR paramétrique (hypothèse normalité) :
\text{VaR}_{95\%} = \mu - 1.65 \times \sigma Où 1.65 = quantile 5% de la loi normale standard.
a)	Calculez VaR 95% mensuelle (%) pour A et B
b)	Calculez VaR 95% annuelle (%) : \text{VaR}_{annuel} = \mu_{annuel} - 1.65 \times \sigma_{annuel}
c)	Convertissez VaR annuelle en perte monétaire (€) sur capital €500,000
d)	Testez l'hypothèse de normalité avec test Shapiro-Wilk (scipy.stats.shapiro)
📌 Interprétation attendue :
 	La VaR annuelle de chaque portefeuille respecte-t-elle la contrainte client (max -€50,000) ?
 	Les données sont-elles compatibles avec loi normale (p-value Shapiro > 0.05) ?


Question 1.4 — Ratio Sharpe et recommandation (6 points)

Calculez le Ratio Sharpe pour évaluer rendement ajusté du risque :
\text{Sharpe} = \frac{R_{annuel} - r_f}{\sigma_{annuel}} Avec r_f = 3\% (taux sans risque OAT 10 ans).
a)	Calculez Sharpe pour A et B
b)	Recommandation client : Quel portefeuille recommandez-vous selon critères :
  VaR ≤ -10% (€50,000)
 	Sharpe maximum
 	Normalité rendements Justifiez en 3-5 phrases.
 
 
✅ CORRECTION PARTIE 1
import numpy as np import pandas as pd
import matplotlib.pyplot as plt import seaborn as sns
from scipy import stats

# Configuration sns.set_style("whitegrid") plt.rcParams['figure.figsize'] = (14, 6)

print("="*80)
print("TP PARTIE 1 — STATISTIQUES ET LOI NORMALE EN FINANCE")
print("Analyse risque portefeuille et calcul VaR") print("="*80)

# ============================================================================ # DONNÉES
# ============================================================================

# Rendements mensuels historiques (%) rendements_A = np.array([
1.2, 0.8, -0.5, 1.5, 0.9, 1.1, 0.7, 1.3, 1.0, 0.6, 1.4, 0.8,
1.1, 0.9, -0.3, 1.2, 1.0, 1.5, 0.8, 1.3, 0.9, 1.1, 1.2, 1.0
])

rendements_B = np.array([
4.5, -2.1, 6.2, -3.5, 5.8, 7.1, -1.8, 4.9, 3.2, -4.2, 8.5, -2.7,
5.1, 6.8, -3.1, 7.3, 4.5, -2.9, 6.7, 5.3, -3.8, 7.9, 4.2, 5.5
])

# Paramètres
capital = 500000 # € à investir perte_max_toleree = 50000 # € (10% capital) taux_sans_risque = 3.0 # % annuel

# ============================================================================ # QUESTION 1.1 — STATISTIQUES DESCRIPTIVES
# ============================================================================

print("\n" + "="*80)
print("QUESTION 1.1 — STATISTIQUES DESCRIPTIVES")
print("="*80)

def calculer_stats_portefeuille(rendements, nom): """
Calcule statistiques descriptives portefeuille

Parameters:
 
rendements : np.array Rendements mensuels (%)
nom : str
Nom portefeuille Returns:
dict : Statistiques calculées """
# a) Moyenne mensuelle moyenne_mensuelle = np.mean(rendements)

# b) Écart-type mensuel
ecart_type_mensuel = np.std(rendements, ddof=1) # ddof=1 pour échantillon

# c) Médiane
mediane = np.median(rendements)

# d) Rendement annualisé (capitalisation composée) # Formule : (1 + r_mensuel/100)^12 - 1
rendement_annuel = ((1 + moyenne_mensuelle/100)**12 - 1) * 100

# e) Volatilité annualisée
# Formule : σ_annuel = σ_mensuel × √12 volatilite_annuelle = ecart_type_mensuel * np.sqrt(12)

stats = {
'nom': nom,
'moyenne_mensuelle': moyenne_mensuelle, 'ecart_type_mensuel': ecart_type_mensuel, 'mediane': mediane,
'rendement_annuel': rendement_annuel, 'volatilite_annuelle': volatilite_annuelle
}

return stats

# Calcul stats pour les deux portefeuilles
stats_A = calculer_stats_portefeuille(rendements_A, "CONSERVATIVE (A)") stats_B = calculer_stats_portefeuille(rendements_B, "AGRESSIF (B)")

# Affichage résultats
for stats in [stats_A, stats_B]:
print(f"\n📊 PORTEFEUILLE {stats['nom']}")
print(f"	• Rendement mensuel moyen : {stats['moyenne_mensuelle']:.2f}%") print(f"	• Écart-type mensuel : {stats['ecart_type_mensuel']:.2f}%") print(f"	• Médiane : {stats['mediane']:.2f}%")
print(f"	• Rendement annualisé : {stats['rendement_annuel']:.2f}%") print(f"	• Volatilité annualisée : {stats['volatilite_annuelle']:.2f}%")

# ============================================================================ # QUESTION 1.2 — VISUALISATION DISTRIBUTIONS
# ============================================================================

print("\n" + "="*80)
print("QUESTION 1.2 — VISUALISATION DISTRIBUTIONS")
print("="*80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Subplot 1 : Histogrammes superposés ax1 = axes[0]
ax1.hist(rendements_A, bins=10, alpha=0.6, color='green', edgecolor='black', label='Portefeuille A (Conservative)', density=True)
ax1.hist(rendements_B, bins=10, alpha=0.6, color='red', edgecolor='black', label='Portefeuille B (Agressif)', density=True)

# Lignes moyennes
ax1.axvline(stats_A['moyenne_mensuelle'], color='darkgreen', linestyle='--', linewidth=2, label=f'Moyenne A = {stats_A["moyenne_mensuelle"]:.2f}%')
ax1.axvline(stats_B['moyenne_mensuelle'], color='darkred', linestyle='--', linewidth=2, label=f'Moyenne B = {stats_B["moyenne_mensuelle"]:.2f}%')

ax1.set_title('Distributions rendements mensuels', fontsize=12, fontweight='bold') ax1.set_xlabel('Rendement mensuel (%)')
ax1.set_ylabel('Densité') ax1.legend(fontsize=9) ax1.grid(True, alpha=0.3)

# Subplot 2 : Boxplots comparatifs ax2 = axes[1]
data_boxplot = [rendements_A, rendements_B]
bp = ax2.boxplot(data_boxplot, labels=['Portefeuille A', 'Portefeuille B'], patch_artist=True, widths=0.6)

# Couleurs boxplots
colors = ['lightgreen', 'lightcoral']
for patch, color in zip(bp['boxes'], colors): patch.set_facecolor(color)

ax2.set_title('Boxplots comparatifs (outliers visibles)', fontsize=12, fontweight='bold') ax2.set_ylabel('Rendement mensuel (%)')
ax2.grid(True, alpha=0.3, axis='y')
ax2.axhline(0, color='black', linestyle=':', linewidth=1)

plt.tight_layout() plt.show()
print("✓ Graphiques générés (histogrammes + boxplots)")

# ============================================================================ # QUESTION 1.3 — VALUE AT RISK (VaR 95%)
# ============================================================================

print("\n" + "="*80)
print("QUESTION 1.3 — VALUE AT RISK (VaR 95%)")
print("="*80)

def calculer_var_portefeuille(stats_dict, capital, alpha=0.05): """
Calcule VaR paramétrique mensuelle et annuelle Parameters:
stats_dict : dict
Statistiques portefeuille (from calculer_stats_portefeuille) capital : float
Capital investi (€) alpha : float
Niveau risque (0.05 pour VaR 95%) Returns:
dict : VaR calculées """
# Quantile normal standard pour alpha=5% (queue gauche) z_alpha = stats.norm.ppf(alpha) # ≈ -1.645

# a) VaR mensuelle (%)
var_mensuelle_pct = stats_dict['moyenne_mensuelle'] + z_alpha * stats_dict['ecart_type_mensuel']

# b) VaR annuelle (%)
# Méthode : Utiliser rendement et volatilité annualisés
var_annuelle_pct = stats_dict['rendement_annuel'] + z_alpha * stats_dict['volatilite_annuelle']

# c) VaR en perte monétaire (€)
var_mensuelle_euros = capital * (var_mensuelle_pct / 100) var_annuelle_euros = capital * (var_annuelle_pct / 100)

var_results = {
'var_mensuelle_pct': var_mensuelle_pct, 'var_annuelle_pct': var_annuelle_pct, 'var_mensuelle_euros': var_mensuelle_euros, 'var_annuelle_euros': var_annuelle_euros
}

return var_results

# Calcul VaR pour les deux portefeuilles
var_A = calculer_var_portefeuille(stats_A, capital) var_B = calculer_var_portefeuille(stats_B, capital)

# Affichage résultats
print(f"\n💰 CAPITAL INVESTI : €{capital:,.0f}")
print(f"🚨 PERTE MAX TOLÉRÉE CLIENT : €{perte_max_toleree:,.0f} (-{perte_max_toleree/capital*100:.0f}%)")
print(f"\n📉 PORTEFEUILLE A (Conservative)")
print(f"	• VaR 95% mensuelle : {var_A['var_mensuelle_pct']:.2f}% → €{var_A['var_mensuelle_euros']:,.0f}") print(f"	• VaR 95% annuelle : {var_A['var_annuelle_pct']:.2f}% → €{var_A['var_annuelle_euros']:,.0f}")
print(f"\n📉 PORTEFEUILLE B (Agressif)")
print(f"	• VaR 95% mensuelle : {var_B['var_mensuelle_pct']:.2f}% → €{var_B['var_mensuelle_euros']:,.0f}") print(f"	• VaR 95% annuelle : {var_B['var_annuelle_pct']:.2f}% → €{var_B['var_annuelle_euros']:,.0f}")

# Vérification contrainte client
print(f"\n✅ VALIDATION CONTRAINTE CLIENT (VaR annuelle ≤ -€50,000) :")
contrainte_A = abs(var_A['var_annuelle_euros']) <= perte_max_toleree contrainte_B = abs(var_B['var_annuelle_euros']) <= perte_max_toleree
print(f"	• Portefeuille A : {'✓ RESPECTÉE' if contrainte_A else '✗ NON RESPECTÉE'} " f"({var_A['var_annuelle_euros']:,.0f} € vs -{perte_max_toleree:,.0f} €)")
print(f"	• Portefeuille B : {'✓ RESPECTÉE' if contrainte_B else '✗ NON RESPECTÉE'} "
f"({var_B['var_annuelle_euros']:,.0f} € vs -{perte_max_toleree:,.0f} €)")

# d) Test normalité (Shapiro-Wilk)
print(f"\n🔬 TEST NORMALITÉ (Shapiro-Wilk, H0: données normales)")

stat_A, p_value_A = stats.shapiro(rendements_A) stat_B, p_value_B = stats.shapiro(rendements_B)

print(f"\n	PORTEFEUILLE A :")
print(f"	• Statistique Shapiro : {stat_A:.4f}") print(f"	• P-value : {p_value_A:.4f}")
if p_value_A > 0.05:
print(f"	✓ Données compatibles loi normale (p > 0.05)") else:
print(f"	✗ Données s'écartent loi normale (p < 0.05) → VaR paramétrique moins fiable")

print(f"\n	PORTEFEUILLE B :")
print(f"	• Statistique Shapiro : {stat_B:.4f}") print(f"	• P-value : {p_value_B:.4f}")
if p_value_B > 0.05:
print(f"	✓ Données compatibles loi normale (p > 0.05)") else:
print(f"	✗ Données s'écartent loi normale (p < 0.05) → VaR paramétrique moins fiable")

# ============================================================================ # QUESTION 1.4 — RATIO SHARPE ET RECOMMANDATION
# ============================================================================

print("\n" + "="*80)
print("QUESTION 1.4 — RATIO SHARPE ET RECOMMANDATION CLIENT")
print("="*80)

# a) Calcul Ratio Sharpe
sharpe_A = (stats_A['rendement_annuel'] - taux_sans_risque) / stats_A['volatilite_annuelle'] sharpe_B = (stats_B['rendement_annuel'] - taux_sans_risque) / stats_B['volatilite_annuelle']
print(f"\n📊 RATIO SHARPE (Rendement ajusté risque)")

print(f"	Formule : (Rendement annuel - Taux sans risque) / Volatilité annuelle")
print(f"	Taux sans risque (rf) : {taux_sans_risque}%")

print(f"\n	PORTEFEUILLE A :")
print(f"	• Sharpe = ({stats_A['rendement_annuel']:.2f} - {taux_sans_risque}) / {stats_A['volatilite_annuelle']:.2f}") print(f"	• Sharpe = {sharpe_A:.3f}")

print(f"\n	PORTEFEUILLE B :")
print(f"	• Sharpe = ({stats_B['rendement_annuel']:.2f} - {taux_sans_risque}) / {stats_B['volatilite_annuelle']:.2f}") print(f"	• Sharpe = {sharpe_B:.3f}")

# Interprétation Sharpe print(f"\n	INTERPRÉTATION :")
if sharpe_A > 1:
print(f"	✓ Portefeuille A : Excellent (Sharpe > 1)") elif sharpe_A > 0.5:
print(f"	✓ Portefeuille A : Bon (0.5 < Sharpe < 1)") else:
print(f"	✗ Portefeuille A : Faible (Sharpe < 0.5)")

if sharpe_B > 1:
print(f"	✓ Portefeuille B : Excellent (Sharpe > 1)") elif sharpe_B > 0.5:
print(f"	✓ Portefeuille B : Bon (0.5 < Sharpe < 1)") else:
print(f"	✗ Portefeuille B : Faible (Sharpe < 0.5)")

# b) RECOMMANDATION CLIENT
print(f"\n" + "="*80)
print("🎯 RECOMMANDATION CLIENT FINALE")
print("="*80)
print(f"\n📋 CRITÈRES DÉCISION :")
print(f"	1. VaR 95% annuelle ≤ -€50,000 (contrainte risque)") print(f"	2. Ratio Sharpe maximum (efficience)")
print(f"	3. Normalité rendements (fiabilité VaR)")
print(f"\n📊 TABLEAU COMPARATIF :")
print(f"\n{'Critère':<30} {'Portefeuille A':<20} {'Portefeuille B':<20}") print(f"{'-'*70}")
print(f"{'Rendement annuel':<30} {stats_A['rendement_annuel']:>8.2f}% {stats_B['rendement_annuel']:>28.2f}%") print(f"{'Volatilité annuelle':<30} {stats_A['volatilite_annuelle']:>8.2f}% {stats_B['volatilite_annuelle']:>28.2f}%") print(f"{'VaR 95% (€)':<30} {var_A['var_annuelle_euros']:>13,.0f} € {var_B['var_annuelle_euros']:>22,.0f} €")
print(f"{'Contrainte respectée':<30} {'✓ OUI' if contrainte_A else '✗ NON':<20} {'✓ OUI' if contrainte_B else '✗ NON':<20}")
print(f"{'Ratio Sharpe':<30} {sharpe_A:>13.3f} {sharpe_B:>27.3f}") print(f"{'Normalité (p-value)':<30} {p_value_A:>13.3f} {p_value_B:>27.3f}")
print(f"\n💡 RECOMMANDATION FINALE :")

# Logique décision
if not contrainte_A and not contrainte_B:
print(f"	❌ AUCUN PORTEFEUILLE ne respecte contrainte risque client.") print(f"	→ Réduire allocation ou revoir tolérance perte.")
elif contrainte_A and not contrainte_B:
print(f"	✅ PORTEFEUILLE A (Conservative) RECOMMANDÉ") print(f"	→ Seul respecte VaR ≤ -€50,000")
print(f"	→ Sharpe {sharpe_A:.2f} correct, volatilité maîtrisée {stats_A['volatilite_annuelle']:.1f}%") elif not contrainte_A and contrainte_B:
print(f"	✅ PORTEFEUILLE B (Agressif) RECOMMANDÉ")
print(f"	→ Seul respecte VaR ≤ -€50,000")
print(f"	→ Rendement élevé {stats_B['rendement_annuel']:.1f}% mais volatilité importante {stats_B['volatilite_annuelle']:.1f}% else: # Les deux respectent contrainte
if sharpe_A > sharpe_B:
print(f"	✅ PORTEFEUILLE A (Conservative) RECOMMANDÉ")
print(f"	→ Meilleur Sharpe ({sharpe_A:.2f} vs {sharpe_B:.2f})") print(f"	→ Profil risque/rendement optimal selon contrainte client")
else:
print(f"	✅ PORTEFEUILLE B (Agressif) RECOMMANDÉ")
print(f"	→ Meilleur Sharpe ({sharpe_B:.2f} vs {sharpe_A:.2f})")
print(f"	→ Rendement supérieur ({stats_B['rendement_annuel']:.1f}% vs {stats_A['rendement_annuel']:.1f}%)")

print(f"\n	JUSTIFICATION :")
print(f"	• VaR paramétrique fiable si normalité vérifiée (test Shapiro p > 0.05)") print(f"	• Sharpe mesure efficience : unités rendement excédentaire par unité risque") print(f"	• Client conservateur → Privilégier A (stabilité)")
print(f"	• Client tolérant volatilité → Envisager B si Sharpe meilleur et VaR OK") print(f"\n✓ FIN PARTIE 1\n")
 

 
poPARTIE 2 — THÉORÈME DE BAYES ET SCORING CRÉDIT
Mise à jour probabilités risque avec nouvelles informations
Durée estimée : 45 minutes | Points : 30/100

📝 ÉNONCÉ PARTIE 2
Contexte métier
Vous êtes data analyst dans le département risques d'une banque retail. Votre mission : construire un système de scoring crédit dynamique utilisant le théorème de Bayes pour mettre à jour la probabilité de défaut d'un emprunteur en fonction de nouveaux événements (retards paiements, découverts bancaires).

Données initiales (Prior)

 	Taux défaut base (toute population) : P(\text{Défaut}) = 5\%
 	Segmentation clients :
 	Segment Premium (30% clients) : Taux défaut 1.5%
 	Segment Standard (50% clients) : Taux défaut 5%
 	Segment Risque (20% clients) : Taux défaut 15%

Événements observables (Likelihood)

Événement	P(Événement | Défaut)	P(Événement | Non-défaut)
Retard paiement	80%	10%
Découvert >500€	65%	15%
Demande crédit refusée ailleurs	55%	8%
QUESTIONS (Partie 2)

Question 2.1 — Calcul Bayes manuel (10 points)

Un client du segment Standard (prior défaut = 5%) présente un retard de paiement ce mois.
a)	Calculez P(\text{Défaut} | \text{Retard}) avec le théorème de Bayes (détaillez calculs)
b)	Interprétez le résultat : de combien augmente le risque ? (facteur multiplicatif)
c)	Quelle décision métier recommandez-vous ? (Surveillance, restriction crédit, maintien conditions)
📌 Formule Bayes :
P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B|A) \cdot P(A) + P(B|\neg A) \cdot P(\neg A)}


Question 2.2 — Mise à jour séquentielle (8 points)

Le même client présente 2 semaines après un découvert >500€.
a)	Utilisez la probabilité posterior Question 2.1 comme nouveau priorb) Calculez P(\text{Défaut} | \text{Retard ET Découvert})c) Tracez un graphique évolution probabilité défaut :
 	Axe X : Étapes (0=Prior, 1=Après Retard, 2=Après Découvert)
 	Axe Y : Probabilité défaut (%)
 	Type : Line plot avec markers


Question 2.3 — Fonction générique Bayes (8 points)

Créez une fonction Python bayes_update(prior, likelihood_pos, likelihood_neg) qui :

 	Inputs : Prior, P(Evidence|Positive), P(Evidence|Negative)
 	Output : Posterior P(Positive|Evidence)
 	Docstring complète avec exemple d'usage
Testez avec les 3 événements sur un client Segment Risque (prior 15%).


Question 2.4 — Matrice confusion et lien Bayes (4 points)

Sur 10,000 clients testés avec modèle retard-paiement :
 	500 défauts réels (5%)
 	400 vrais positifs détectés (TP)
 	950 faux positifs (FP)
a)	Calculez Precision = TP/(TP+FP)
b)	Vérifiez cohérence avec P(\text{Défaut}|\text{Retard}) Question 2.1
c)	Expliquez pourquoi Precision bayésienne = Precision matrice confusion


 
✅ CORRECTION PARTIE 2
import numpy as np import pandas as pd
import matplotlib.pyplot as plt import seaborn as sns

# Configuration sns.set_style("whitegrid")

print("="*80)
print("TP PARTIE 2 — THÉORÈME DE BAYES ET SCORING CRÉDIT")
print("Mise à jour probabilités risque avec nouvelles informations") print("="*80)

# ============================================================================ # DONNÉES
# ============================================================================

# Taux défaut base et segmentation taux_defaut_base = 0.05 # 5%

segments = {
'Premium': {'proportion': 0.30, 'taux_defaut': 0.015},
'Standard': {'proportion': 0.50, 'taux_defaut': 0.05},
'Risque': {'proportion': 0.20, 'taux_defaut': 0.15}
}

# Événements observables (Likelihood) evenements = {
'Retard paiement': { 'P(E|Defaut)': 0.80,
'P(E|Non-defaut)': 0.10
},
'Decouvert >500€': { 'P(E|Defaut)': 0.65,
'P(E|Non-defaut)': 0.15
},
'Refus credit ailleurs': { 'P(E|Defaut)': 0.55,
'P(E|Non-defaut)': 0.08
}
}

# ============================================================================ # QUESTION 2.1 — CALCUL BAYES MANUEL
# ============================================================================

print("\n" + "="*80)
print("QUESTION 2.1 — CALCUL BAYES MANUEL")
print("="*80)
print("\n📋 CONTEXTE :")
print("	Client Segment Standard présente un RETARD PAIEMENT")
print(f"	• Prior P(Défaut) = {segments['Standard']['taux_defaut']:.1%}")
print(f"	• P(Retard|Défaut) = {evenements['Retard paiement']['P(E|Defaut)']:.0%}") print(f"	• P(Retard|Non-défaut) = {evenements['Retard paiement']['P(E|Non-defaut)']:.0%}")

# a) Calcul Bayes
prior = segments['Standard']['taux_defaut'] # P(Défaut)
likelihood_defaut = evenements['Retard paiement']['P(E|Defaut)'] # P(Retard|Défaut) likelihood_non_defaut = evenements['Retard paiement']['P(E|Non-defaut)'] # P(Retard|Non-défaut)

# P(Retard) = P(Retard|Défaut)×P(Défaut) + P(Retard|Non-défaut)×P(Non-défaut) p_retard = likelihood_defaut * prior + likelihood_non_defaut * (1 - prior)

# P(Défaut|Retard) = P(Retard|Défaut) × P(Défaut) / P(Retard) posterior = (likelihood_defaut * prior) / p_retard
print(f"\n🧮 CALCUL DÉTAILLÉ BAYES :")
print(f"\n	Étape 1 : Calcul P(Retard) via loi probabilités totales")
print(f"	P(Retard) = P(Retard|Défaut)×P(Défaut) + P(Retard|Non-défaut)×P(Non-défaut)")
print(f"	P(Retard) = {likelihood_defaut:.2f} × {prior:.2f} + {likelihood_non_defaut:.2f} × {1-prior:.2f}") print(f"	P(Retard) = {likelihood_defaut * prior:.4f} + {likelihood_non_defaut * (1-prior):.4f}")
print(f"	P(Retard) = {p_retard:.4f} = {p_retard:.2%}")

print(f"\n	Étape 2 : Théorème de Bayes")
print(f"	P(Défaut|Retard) = P(Retard|Défaut) × P(Défaut) / P(Retard)")
print(f"	P(Défaut|Retard) = {likelihood_defaut:.2f} × {prior:.2f} / {p_retard:.4f}") print(f"	P(Défaut|Retard) = {likelihood_defaut * prior:.4f} / {p_retard:.4f}") print(f"	P(Défaut|Retard) = {posterior:.4f} = {posterior:.2%}")

# b) Interprétation facteur multiplicatif facteur_multiplication = posterior / prior
print(f"\n📊 INTERPRÉTATION :")
print(f"	• Prior (avant retard) : {prior:.1%}") print(f"	• Posterior (après retard) : {posterior:.1%}")
print(f"	• Augmentation risque : {(posterior - prior)*100:.1f} points") print(f"	• Facteur multiplication : ×{facteur_multiplication:.2f}")
print(f"	→ Retard paiement MULTIPLIE risque défaut par {facteur_multiplication:.1f} !")

# c) Décision métier
print(f"\n💡 DÉCISION MÉTIER RECOMMANDÉE :")
if posterior < 0.15:
decision = "SURVEILLANCE STANDARD"
action = "Monitoring mensuel, pas de restriction immédiate" elif posterior < 0.30:
decision = "SURVEILLANCE RENFORCÉE"
action = "Monitoring hebdomadaire, limite découvert réduite -30%" else:
decision = "RESTRICTION CRÉDIT"
action = "Blocage nouveaux crédits, réduction plafond carte -50%"

print(f"	✓ DÉCISION : {decision}") print(f"	✓ ACTION : {action}")
print(f"	✓ JUSTIFICATION : Posterior {posterior:.1%} franchit seuil alerte 15%")

# ============================================================================ # QUESTION 2.2 — MISE À JOUR SÉQUENTIELLE
# ============================================================================

print("\n" + "="*80)
print("QUESTION 2.2 — MISE À JOUR SÉQUENTIELLE")
print("="*80)
print("\n📋 CONTEXTE :")
print("	2 semaines après, le même client présente DÉCOUVERT >500€") print("	→ Utilisation posterior Q2.1 comme nouveau prior")

# a) Nouveau prior = posterior Q2.1 prior_2 = posterior

# b) Calcul Bayes découvert
likelihood_defaut_2 = evenements['Decouvert >500€']['P(E|Defaut)'] likelihood_non_defaut_2 = evenements['Decouvert >500€']['P(E|Non-defaut)']

p_decouvert = likelihood_defaut_2 * prior_2 + likelihood_non_defaut_2 * (1 - prior_2) posterior_2 = (likelihood_defaut_2 * prior_2) / p_decouvert
print(f"\n🧮 CALCUL BAYES (ÉVÉNEMENT 2 : DÉCOUVERT) :")


print(f"	• Nouveau prior P(Défaut) = {prior_2:.4f} (= posterior Q2.1)")
print(f"	• P(Découvert|Défaut) = {likelihood_defaut_2:.0%}") print(f"	• P(Découvert|Non-défaut) = {likelihood_non_defaut_2:.0%}")

print(f"\n		P(Découvert) = {likelihood_defaut_2:.2f} × {prior_2:.4f} + {likelihood_non_defaut_2:.2f} × {1-prior_2:.4f}") print(f"	P(Découvert) = {p_decouvert:.4f}")

print(f"\n		P(Défaut|Retard ET Découvert) = {likelihood_defaut_2:.2f} × {prior_2:.4f} / {p_decouvert:.4f}") print(f"	P(Défaut|Retard ET Découvert) = {posterior_2:.4f} = {posterior_2:.2%}")
print(f"\n📊 ÉVOLUTION PROBABILITÉ DÉFAUT :")
print(f"	Étape 0 (Prior initial) : {prior:.1%}")
print(f"	Étape 1 (Après Retard) : {posterior:.1%} (+{(posterior-prior)*100:.1f} pts)")
print(f"	Étape 2 (Après Découvert) : {posterior_2:.1%} (+{(posterior_2-posterior)*100:.1f} pts)") print(f"	→ TOTAL : ×{posterior_2/prior:.2f} augmentation risque depuis prior initial")

# c) Graphique évolution
fig, ax = plt.subplots(figsize=(10, 6))

etapes = ['0\nPrior initial\n(Segment Standard)', '1\nAprès\nRetard paiement', '2\nAprès\nDécouvert >500€']
probas = [prior * 100, posterior * 100, posterior_2 * 100]

ax.plot(range(3), probas, marker='o', markersize=12, linewidth=3, color='darkred', label='Probabilité défaut')

# Points
for i, (etape, proba) in enumerate(zip(etapes, probas)): ax.annotate(f'{proba:.2f}%', xy=(i, proba), xytext=(0, 10),
textcoords='offset points', ha='center', fontsize=11, fontweight='bold', bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.7))

# Seuils décision
ax.axhline(15, color='orange', linestyle='--', linewidth=2, alpha=0.7, label='Seuil surveillance renforcée (15%)') ax.axhline(30, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Seuil restriction crédit (30%)')

ax.set_xticks(range(3)) ax.set_xticklabels(etapes)
ax.set_ylabel('Probabilité défaut (%)', fontsize=12, fontweight='bold') ax.set_title('Mise à jour séquentielle risque crédit (Théorème Bayes)',
fontsize=13, fontweight='bold') ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout() plt.show()
print("\n✓ Graphique évolution probabilité généré")

# ============================================================================ # QUESTION 2.3 — FONCTION GÉNÉRIQUE BAYES
# ============================================================================

print("\n" + "="*80)
print("QUESTION 2.3 — FONCTION GÉNÉRIQUE BAYES")
print("="*80)

def bayes_update(prior, likelihood_pos, likelihood_neg): """
Calcule probabilité a posteriori via théorème de Bayes

Formule : P(A|B) = P(B|A) × P(A) / P(B) Avec P(B) = P(B|A)×P(A) + P(B|¬A)×P(¬A)

Parameters:

prior : float
Probabilité a priori P(A) ∈ [0, 1] Exemple : 0.05 pour taux défaut 5%

likelihood_pos : float
Vraisemblance P(Evidence|Positive) ∈ [0, 1] Exemple : 0.80 pour P(Retard|Défaut)

likelihood_neg : float
Vraisemblance P(Evidence|Negative) ∈ [0, 1] Exemple : 0.10 pour P(Retard|Non-défaut)

Returns:

posterior : float
Probabilité a posteriori P(A|B) ∈ [0, 1] Raises:
ValueError : Si paramètres hors [0, 1] Examples:
>>> # Client défaut 5%, observe retard (80% si défaut, 10% si sain)
>>> posterior = bayes_update(prior=0.05, likelihood_pos=0.80, likelihood_neg=0.10)
>>> print(f"P(Défaut|Retard) = {posterior:.2%}") P(Défaut|Retard) = 29.63%

>>> # Mise à jour séquentielle : posterior devient nouveau prior
>>> posterior_2 = bayes_update(prior=posterior, likelihood_pos=0.65, likelihood_neg=0.15)
>>> print(f"P(Défaut|Retard ET Découvert) = {posterior_2:.2%}") P(Défaut|Retard ET Découvert) = 55.88%
"""
# Validation inputs
if not (0 <= prior <= 1):
raise ValueError(f"prior doit être dans [0,1], reçu {prior}") if not (0 <= likelihood_pos <= 1):
raise ValueError(f"likelihood_pos doit être dans [0,1], reçu {likelihood_pos}") if not (0 <= likelihood_neg <= 1):

raise ValueError(f"likelihood_neg doit être dans [0,1], reçu {likelihood_neg}")

# Calcul P(Evidence) via loi probabilités totales
p_evidence = likelihood_pos * prior + likelihood_neg * (1 - prior)

# Protection division par zéro if p_evidence == 0:
return 0.0

# Théorème de Bayes
posterior = (likelihood_pos * prior) / p_evidence return posterior
print("\n✓ Fonction bayes_update() créée avec docstring complète")

# Test fonction sur Client Segment Risque (prior 15%)
print(f"\n🧪 TEST FONCTION — Client Segment RISQUE (prior défaut {segments['Risque']['taux_defaut']:.0%})")

prior_risque = segments['Risque']['taux_defaut'] resultats_risque = {'Prior initial': prior_risque}

# Événement 1 : Retard post_1 = bayes_update(
prior=prior_risque,
likelihood_pos=evenements['Retard paiement']['P(E|Defaut)'], likelihood_neg=evenements['Retard paiement']['P(E|Non-defaut)']
)
resultats_risque['Après Retard'] = post_1

# Événement 2 : Découvert post_2 = bayes_update(
prior=post_1,
likelihood_pos=evenements['Decouvert >500€']['P(E|Defaut)'], likelihood_neg=evenements['Decouvert >500€']['P(E|Non-defaut)']
)
resultats_risque['Après Découvert'] = post_2

# Événement 3 : Refus crédit post_3 = bayes_update(
prior=post_2,
likelihood_pos=evenements['Refus credit ailleurs']['P(E|Defaut)'], likelihood_neg=evenements['Refus credit ailleurs']['P(E|Non-defaut)']
)
resultats_risque['Après Refus crédit'] = post_3
print(f"\n📊 RÉSULTATS TEST (Client Segment Risque) :") for etape, proba in resultats_risque.items():
print(f"	{etape:<25} : P(Défaut) = {proba:.4f} ({proba:.2%})")
print(f"\n💡 INTERPRÉTATION :")
print(f"	• Risque initial : {prior_risque:.0%} (segment risque)") print(f"	• Après 3 événements négatifs : {post_3:.0%}")
print(f"	• Multiplication risque : ×{post_3/prior_risque:.1f}")
print(f"	→ Client très haut risque, recommandation : REJET crédit ou garanties renforcées")

# ============================================================================ # QUESTION 2.4 — MATRICE CONFUSION ET LIEN BAYES
# ============================================================================

print("\n" + "="*80)
print("QUESTION 2.4 — MATRICE CONFUSION ET LIEN BAYES")
print("="*80)
print("\n📋 DONNÉES MATRICE CONFUSION (10,000 clients testés) :")

n_total = 10000
n_defauts_reels = 500 # 5% taux défaut n_non_defauts_reels = n_total - n_defauts_reels

tp = 400 # Vrais positifs (défauts détectés par retard) fp = 950 # Faux positifs (non-défauts avec retard)
fn = n_defauts_reels - tp # Faux négatifs
tn = n_non_defauts_reels - fp # Vrais négatifs

print(f"	• Total clients : {n_total:,}")
print(f"	• Défauts réels : {n_defauts_reels} ({n_defauts_reels/n_total:.0%})") print(f"	• Vrais positifs (TP) : {tp}")
print(f"	• Faux positifs (FP) : {fp}") print(f"	• Faux négatifs (FN) : {fn}") print(f"	• Vrais négatifs (TN) : {tn}")

print(f"\n	MATRICE CONFUSION")
print(f"		RÉALITÉ") print(f"	Non-défaut Défaut")
print(f"PRÉD Retard		{fp:4d}		{tp:3d}") print(f"	Pas	{tn:4d}	{fn:3d}")

# a) Calcul Precision precision = tp / (tp + fp)
print(f"\n🧮 CALCUL PRECISION :") print(f"	Precision = TP / (TP + FP)")
print(f"	Precision = {tp} / ({tp} + {fp})") print(f"	Precision = {tp} / {tp + fp}")
print(f"	Precision = {precision:.4f} = {precision:.2%}")

# b) Comparaison avec Bayes Q2.1 print(f"\n🔗 COMPARAISON AVEC BAYES Q2.1 :")
print(f"	• P(Défaut|Retard) calculé Bayes : {posterior:.4f} ({posterior:.2%})") print(f"	• Precision matrice confusion : {precision:.4f} ({precision:.2%})")
print(f"	• Différence : {abs(posterior - precision):.4f} ({abs(posterior - precision)*100:.2f} pts)")

if abs(posterior - precision) < 0.01:

print(f"	✓ COHÉRENCE PARFAITE (< 1 pt différence)")
else:
print(f"	⚠ Petite différence due arrondis ou données simulées")

# c) Explication lien Bayes / Precision print(f"\n💡 EXPLICATION LIEN BAYES ↔ PRECISION :")
print(f"""
THÉORÈME DE BAYES :
P(Défaut|Retard) = "Parmi clients avec RETARD, quelle % sont DÉFAUTS réels ?"

PRECISION (Matrice confusion) :
Precision = TP / (TP + FP) = "Parmi prédictions POSITIVES, quelle % correctes ?"

→ ÉQUIVALENCE MATHÉMATIQUE :
Précision mesure P(Classe vraie | Prédiction positive)
= P(Défaut réel | Retard détecté)
= Calcul bayésien de posterior !

→ En ML, optimiser Precision = maximiser probabilités a posteriori bayésiennes
→ Naive Bayes Classifier utilise explicitement P(Classe|Features) via Bayes """)
print(f"\n✓ FIN PARTIE 2\n")


 
PARTIE 3 — K-NEAREST NEIGHBORS ET ÉVALUATION MODÈLE
Classification crédit et optimisation hyperparamètres
Durée estimée : 90 minutes | Points : 40/100

📝 ÉNONCÉ PARTIE 3
Contexte métier
Vous êtes data scientist dans une fintech. Mission : construire un modèle KNN de scoring crédit pour automatiser décisions d'octroi prêts personnels (€5K-€50K). Objectif business :
Maximiser Recall (détecter 80%+ défauts) tout en maintenant Precision >60% Optimiser K via validation croisée
Interpréter ROC/AUC et calculer ROI selon coûts métier

Dataset synthétique fourni
Le code génère un dataset credit_data.csv (2000 clients, 85/15% imbalance) avec features :
age : Âge (25-65 ans)
salaire : Salaire annuel (€20K-€120K) anciennete_emploi : Ancienneté emploi actuel (0-30 ans) dette_totale : Dette totale (€0-€80K) ratio_dette_revenu : Dette/Salaire (0-2) nb_credits_actifs : Nombre crédits en cours (0-5)
historique_retards : Nombre retards 24 derniers mois (0-10) score_credit_bureau : Score bureau crédit (300-850, style FICO) defaut : Target binaire (1=défaut, 0=remboursé)

Coûts métier (pour ROI)

Perte si défaut non détecté (FN) : €15,000 (perte moyenne principal + intérêts)
Coût analyse approfondie (FP) : €500 (vérification manuelle dossier)
Gain si défaut détecté (TP) : €15,000 (perte évitée)
Coût opportunité refus bon client (FP) : €1,200 (marge perdue prêt non accordé)


QUESTIONS (Partie 3)

Question 3.1 — Génération et exploration dataset (8 points)

a) Générez le dataset avec le code fourni ci-dessous (seed=42 pour reproductibilité)b) Affichez 5 premières lignes et statistiques descriptives (df.describe())c) Calculez et affichez :
Taux défaut (%) : defaut.mean()
Distribution classes : defaut.value_counts()
Corrélation features vs target : df.corr()['defaut'].sort_values()
d)	Créez 2 visualisations :
Heatmap corrélation (seaborn sns.heatmap)
Boxplots 2 features les plus corrélées avec défaut (ex: ratio_dette_revenu, historique_retards)
Code génération dataset :
import numpy as np import pandas as pd

np.random.seed(42)

n_samples = 2000
age = np.random.randint(25, 66, n_samples)
salaire = np.random.normal(50000, 20000, n_samples).clip(20000, 120000) anciennete_emploi = np.random.exponential(5, n_samples).clip(0, 30)

dette_totale = np.random.normal(25000, 15000, n_samples).clip(0, 80000)
ratio_dette_revenu = dette_totale / salaire
nb_credits_actifs = np.random.poisson(1.5, n_samples).clip(0, 5) historique_retards = np.random.poisson(2, n_samples).clip(0, 10) score_credit = np.random.normal(650, 100, n_samples).clip(300, 850)

# Target : Défaut si combo risque élevé defaut_proba = (
0.05 + # Baseline 5%
0.15 * (ratio_dette_revenu > 0.5) +
0.10 * (historique_retards > 3) +
0.08 * (score_credit < 600) +
0.05 * (nb_credits_actifs > 2)
).clip(0, 0.85)

defaut = (np.random.rand(n_samples) < defaut_proba).astype(int)

df = pd.DataFrame({ 'age': age, 'salaire': salaire,
'anciennete_emploi': anciennete_emploi, 'dette_totale': dette_totale, 'ratio_dette_revenu': ratio_dette_revenu, 'nb_credits_actifs': nb_credits_actifs, 'historique_retards': historique_retards, 'score_credit_bureau': score_credit, 'defaut': defaut
})

df.to_csv('credit_data.csv', index=False)
print(f"Dataset généré : {len(df)} clients, taux défaut {defaut.mean():.1%}")


Question 3.2 — Preprocessing et split train/test (6 points)

a)	Séparez features (X) et target (y)
b)	Split train/test 70/30 avec stratify=y (préserve proportion classes)
c)	Normalisez features avec StandardScaler (fit sur train, transform train+test)
d)	Affichez tailles train/test et distribution classes dans chaque set
📌 Librairies : train_test_split, StandardScaler (sklearn)


Question 3.3 — Recherche hyperparamètre K optimal (10 points)

a)	Testez K = 1, 3, 5, 7, 9, 11, 15, 20, 25, 30 avec KNeighborsClassifierb) Pour chaque K, calculez via 5-fold cross-validation sur train set :
 	AUC moyen (métrique principale)
 	Recall moyen
 	Precision moyenne
c)	Stockez résultats dans DataFrame avec colonnes ['K', 'AUC_mean', 'AUC_std', 'Recall_mean', 'Precision_mean']
d)	Identifiez K optimal = K avec AUC maximum
e)	Visualisez : Courbe AUC vs K (avec barres d'erreur = AUC_std)
📌 Fonctions : cross_val_score, make_scorer, roc_auc_score


Question 3.4 — Entraînement modèle final et évaluation (10 points)

a)	Entraînez modèle KNN final avec K optimal sur train set completb) Prédictions test set : Classes (y_pred) et probabilités (y_pred_proba)c) Calculez et affichez matrice de confusion (TP, FP, FN, TN)d) Calculez métriques :
 	Accuracy, Precision, Recall, F1-score
 	AUC-ROC
 	Specificity = TN / (TN + FP)
e)	Affichez classification_report sklearn
f)	Visualisez matrice confusion avec seaborn heatmap (annotations valeurs)


Question 3.5 — Courbe ROC et analyse seuil (6 points)

a)	Tracez courbe ROC (TPR vs FPR) avec AUC dans légende
b)	Ajoutez ligne diagonale (classifieur aléatoire AUC=0.5)
c)	Calculez indice Youden : J = \text{TPR} - \text{FPR} pour trouver seuil optimal
d)	Marquez point optimal sur courbe ROC
e)	Testez 3 seuils (0.3, 0.5, 0.7) : Calculez Precision/Recall/F1 pour chaque
f)	Recommandez seuil métier selon contrainte Recall ≥ 80%
📌 Formule Youden : Seuil qui maximise (Sensibilité + Spécificité - 1)


Question 3.6 — Calcul ROI et recommandation business (5 points bonus)

a)	Avec matrice confusion (seuil 0.5), calculez ROI annuel :
 	Gains détections vraies (TP) : TP × €15,000
 	Coûts analyses (FP) : FP × €500
 	Coûts opportunité (FP) : FP × €1,200
 	Pertes défauts manqués (FN) : FN × €15,000
 	ROI net = Gains - Coûts - Pertes
b)	Calculez ROI pour les 3 seuils testés Q3.5c) Recommandation finale : Quel seuil maximise ROI tout en respectant Recall ≥ 80% ?d) Rédigez executive summary (5-7 phrases) pour direction avec :
 	K optimal choisi et justification
 	Métriques clés (AUC, Recall, Precision)

 	ROI annuel estimé
 	Seuil décision recommandé
 	Impact business attendu

 
✅ CORRECTION PARTIE 3
import numpy as np import pandas as pd
import matplotlib.pyplot as plt import seaborn as sns

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier from sklearn.metrics import (
confusion_matrix, classification_report, roc_curve, roc_auc_score, auc,
precision_score, recall_score, f1_score, accuracy_score, precision_recall_curve
)

# Configuration sns.set_style("whitegrid") plt.rcParams['figure.figsize'] = (12, 5)

print("="*80)
print("TP PARTIE 3 — K-NEAREST NEIGHBORS (KNN) ET ÉVALUATION MODÈLE")
print("Classification crédit et optimisation hyperparamètres") print("="*80)

# ============================================================================

---
# 📚 BARÈME ET RESSOURCES

## Barème détaillé (Total : 100 points)

| **Partie** | **Question** | **Points** | **Critères évaluation** |
|	|	|	|	|
| **Partie 1** (30 pts) | Q1.1 Stats descriptives | 8 pts | Calculs corrects (moyenne, std, annualisation), formules LaTeX si comment
| | Q1.2 Visualisations | 6 pts | Histogrammes + boxplots, labels, légendes, clarté |
| | Q1.3 VaR 95% | 10 pts | Calcul VaR mensuelle/annuelle, test Shapiro, interprétation contrainte client |
| | Q1.4 Ratio Sharpe | 6 pts | Calcul Sharpe A et B, recommandation justifiée (3-5 phrases) |
| **Partie 2** (30 pts) | Q2.1 Bayes manuel | 10 pts | Calcul détaillé étape par étape, interprétation facteur multiplication, décisi
| | Q2.2 Séquentiel | 8 pts | Mise à jour prior→posterior, calcul 2e événement, graphique évolution |
| | Q2.3 Fonction générique | 8 pts | Code générique, docstring complète avec exemple, tests validés |
| | Q2.4 Matrice confusion | 4 pts | Calcul Precision, vérification cohérence Bayes, explication lien conceptuel |
| **Partie 3** (40 pts) | Q3.1 Exploration | 8 pts | Dataset généré, stats descriptives, corrélations, 2 visualisations (heatmap + bo
| | Q3.2 Preprocessing | 6 pts | Split stratifié 70/30, StandardScaler, vérifications normalisation |
| | Q3.3 Optimisation K | 10 pts | CV 5-fold sur K=1-30, DataFrame résultats (AUC, Recall, Precision), graphique AUC vs K |
| | Q3.4 Évaluation | 10 pts | Matrice confusion détaillée (TP/FP/FN/TN), métriques complètes, heatmap, classification_report |
| | Q3.5 ROC | 6 pts | Courbe ROC + AUC, indice Youden, test 3 seuils, recommandation seuil Recall≥80% |
| | Q3.6 ROI (bonus) | 5 pts | Calcul ROI 3 seuils, comparaison, executive summary 5-7 phrases |
| **TOTAL** | | **100 pts** | **+5 bonus** |

---

## Librairies requises

Créez un environnement virtuel et installez les dépendances :

```bash
# Création environnement virtuel python -m venv tp_ia_env
source tp_ia_env/bin/activate # Linux/Mac # ou
tp_ia_env\Scripts\activate # Windows

# Installation librairies pip install numpy==1.24.3 pip install pandas==2.0.3
pip install matplotlib==3.7.2 pip install seaborn==0.12.2
pip install scikit-learn==1.3.0 pip install scipy==1.11.1
Ou avec fichier requirements.txt :

numpy==1.24.3 pandas==2.0.3 matplotlib==3.7.2 seaborn==0.12.2 scikit-learn==1.3.0 scipy==1.11.1
Installation : pip install -r requirements.txt

Fichiers à rendre
1.	Notebook Jupyter (obligatoire)
Nom fichier : TP_IA_Finance_NOM_Prenom.ipynb

Contenu requis :
 	Code Python exécutable (cellules code)
 	Commentaires détaillés expliquant chaque bloc (cellules markdown)
 
Outputs visibles (graphiques, tableaux, métriques)
Pas de cellules vides ou non exécutées Exécution séquentielle de haut en bas sans erreur
Structure recommandée :
# TP INTELLIGENCE ARTIFICIELLE EN FINANCE
**Étudiant** : NOM Prénom
**Date** : JJ/MM/AAAA

## PARTIE 1 — Statistiques et loi normale ### Question 1.1 — Statistiques descriptives [Code + outputs]

### Question 1.2 — Visualisations [Code + graphiques]

... (continuer pour toutes les questions)


2.	Rapport PDF (obligatoire)
Nom fichier : Rapport_TP_IA_NOM_Prenom.pdf

Contenu requis :
Page titre : Nom, prénom, filière, date, titre TP Introduction (0.5 page) : Objectifs TP, contexte métier Réponses aux questions (8-10 pages) :
Formules mathématiques (LaTeX typées ou propres) Captures graphiques (PNG haute résolution) Tableaux résultats (propres, alignés)
Interprétations métier (pas seulement chiffres bruts)
Partie 3 Executive Summary (Question 3.6d, obligatoire) : Recommandation K optimal avec justification Métriques clés (AUC, Recall, Precision)
ROI annuel estimé avec calculs
Seuil décision recommandé et impact business attendu
Conclusion (0.5 page) : Apprentissages clés, difficultés rencontrées, applications métier possibles Références : Cours, documentation Python/sklearn utilisée
Format :
Police : Arial ou Times New Roman, 11-12pt Marges : 2.5cm de chaque côté
Interligne : 1.15 ou 1.5 Numérotation pages
10-15 pages maximum (hors annexes)


Critères notation détaillés
Partie Code (60%)
✅ Exécution sans erreur (10%) : Code tourne de bout en bout
✅ Exactitude calculs (25%) : Résultats numériques corrects, formules bien appliquées
✅ Qualité code (15%) : Commentaires, nommage variables, structure, fonctions réutilisables
✅ Visualisations (10%) : Graphiques clairs, légendes, titres, couleurs appropriées

Partie Rapport (40%)
✅ Interprétation métier (15%) : Traduction résultats techniques en insights business
✅ Justifications décisions (10%) : Recommandations argumentées (ex: quel portefeuille choisir, pourquoi ?)
✅ Executive summary P3 (10%) : Synthèse claire pour direction, ROI business mis en avant
✅ Présentation (5%) : Mise en page soignée, orthographe, clarté rédaction


Conseils réussite
Pour chaque partie :
1.	Lisez l'énoncé 2 fois avant de coder
2.	Testez chaque bloc séparément avant de passer au suivant
3.	Vérifiez outputs : Les chiffres sont-ils cohérents ? (ex: taux défaut 5%, pas 50%)
4.	Interprétez systématiquement : "Ce résultat signifie que..." en termes métier

Debugging fréquents :
Erreur dimension arrays : Vérifiez shapes avec .shape, transposez si besoin array.T
Division par zéro : Ajoutez np.clip() ou vérifications if denom != 0 Graphiques vides : Assurez plt.show() ou dans Jupyter %matplotlib inline Métriques étranges : Vérifiez classes prédites : 0 ou 1 ? Probabilités ou labels ?

Gestion temps (3h) :

Partie 1 : 45 min (simpler, warm-up)
Partie 2 : 45 min (calculs Bayes méthodiques)
Partie 3 : 90 min (plus longue, optimisation K + évaluation complète)
Vérification/rapport : 30 min buffer

 
Ressources complémentaires
Documentation officielle

NumPy : https://numpy.org/doc/stable/ Pandas : https://pandas.pydata.org/docs/ Scikit-learn : https://scikit-learn.org/stable/
Matplotlib : https://matplotlib.org/stable/contents.html
Seaborn : https://seaborn.pydata.org/

Tutoriels KNN

Scikit-learn KNN guide : https://scikit-learn.org/stable/modules/neighbors.html Cross-validation : https://scikit-learn.org/stable/modules/cross_validation.html Metrics : https://scikit-learn.org/stable/modules/model_evaluation.html

Lectures théoriques

VaR (Value at Risk) : Concepts Finance quantitative Théorème de Bayes : Statistiques bayésiennes Ratio Sharpe : Mesures performance portefeuille ROC/AUC : Évaluation classificateurs binaires

Support et aide
Pendant le TP

Questions sur compréhension énoncé : Lever la main
Bugs techniques (installation, imports) : Assistance technique
Interprétations métier : Relire sections contexte, réfléchir en termes Finance/Audit

Après le TP (devoirs)
Forum cours en ligne : Poser questions générales
Email professeur : abderrahim.Larhlimi@uhp.ac.ma (questions spécifiques, délais exceptionnels) Heures permanence : Voir planning cours

⚠ Règles académiques
Intégrité
✅ Autorisé : Consulter cours, documentation officielle, tutoriels en ligne
✅ Autorisé : Discuter approche générale avec camarades ("Comment calcule-t-on VaR ?")
❌ Interdit : Copier/coller code camarades ou Internet sans compréhension
❌ Interdit : Utiliser ChatGPT/LLM pour générer code complet (détection automatique)
❌ Interdit : Soumettre travail identique à plusieurs étudiants

Détection plagiat
Code comparé automatiquement entre tous les rendus
Similarité >70% → Convocation, potentiel 0/100 pour les deux parties

Cas force majeure

Maladie, problème technique grave : Contacter professeur avant deadline avec justificatif Extension délai possible uniquement si demandée à l'avance

📅 Calendrier et modalités remise
Date limite rendu
Deadline : [À COMPLÉTER PAR PROFESSEUR]
Plateforme : [Moodle / Email / Autre]

Format archives
Créez un fichier ZIP nommé : TP_IA_NOM_Prenom.zip contenant :
TP_IA_NOM_Prenom.zip
├── TP_IA_Finance_NOM_Prenom.ipynb
├── Rapport_TP_IA_NOM_Prenom.pdf
└── (optionnel) credit_data.csv (si généré séparément)

Pénalités retard

Retard 0-24h : -10% note finale Retard 24-48h : -20% note finale
Retard >48h : 0/100 (sauf cas force majeure justifié)



🎯 Critères excellence (>90/100)
Pour viser note excellente :

1.	✅ Code impeccable : Fonctions génériques réutilisables, docstrings complètes
2.	✅ Visualisations professionnelles : Graphiques dignes présentation client
3.	✅ Interprétations poussées : Analyse sensibilité (ex: "Si K change de 5 à 7, AUC varie de...")
4.	✅ Executive summary remarquable : Synthèse concise, chiffrée, actionnables pour direction
5.	✅ Bonus Q3.6 : ROI détaillé avec recommandations business concrètes
6.	✅ Dépassement attendu : Tests supplémentaires (ex: comparer KNN vs Logistic Regression complet)

📧 Contact
Professeur : A. Larhlimi
Email : abderrahim.Larhlimi@uhp.ac.ma
Établissement : ENCG Settat
Cours : Intelligence Artificielle - Finance, Contrôle, Audit et Conseil

Bon courage et bon TP ! 🚀
