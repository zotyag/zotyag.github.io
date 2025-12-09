
import numpy as np  # Numerical Python library
import matplotlib.pyplot as plt  # Matlab-like Python module
import seaborn as sns  # Statistical Data Visualization
import pandas as pd  # Data structure and analysis tools (needed for seaborn pairplot)
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits  # Digits loader
from sklearn.decomposition import PCA  # Principal Component Analysis
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, davies_bouldin_score



# Olvassa be a digits beépített adatállományt és írassa ki a legfontosabb jellemzőit
# (rekordok száma, attribútumok száma és osztályok száma). (3 pont)

digits = load_digits()

# --- 1. Data Loading and Setup ---
digits = load_digits()
X = digits.data  # Features (64 attributes)
y = digits.target  # Target (0-9 classes)
n = X.shape[0]  # number of records
p = X.shape[1]  # number of attributes (64)
k = len(np.unique(y))  # number of target classes (10)

print(f'Number of records: {n}')
print(f'Number of attributes: {p}')
print(f'Number of target classes: {k}')



# Készítsen többdimenziós vizualizációt a mátrix ábra segítségével (pairplot). (4 pont)

# --- 2. Dimensionality Reduction using PCA ---
# We reduce the 64 dimensions to 4 Principal Components (PCs) for visualization.
n_components = 4
pca = PCA(n_components=n_components)
X_pca = pca.fit_transform(X)

# --- 3. Prepare Data for Pairplot (using Pandas DataFrame) ---
# Seaborn's pairplot works best with a Pandas DataFrame
column_names = [f'PC{i + 1}' for i in range(n_components)]
pca_df = pd.DataFrame(data=X_pca, columns=column_names)
pca_df['Target'] = y.astype(str)  # Convert target to string for discrete color mapping

# --- 4. Generate Matrix Scatterplot (Pairplot) ---
sns.set(style="ticks")
# The 'hue="Target"' parameter colors the points according to their class (0 to 9)
pairplot_fig = sns.pairplot(pca_df,
                            hue="Target",
                            palette="Spectral",  # A color palette suitable for 10 classes
                            plot_kws={'s': 5, 'alpha': 0.7})  # Adjust point size and transparency
plt.show()



# Particionálja az adatállományt 80% tanító és 20% tesztállományra.
# Keverje össze a rekordokat és a véletlenszám-generátort inicializálja az idei évvel. (3 pont)

current_year = 2025
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,             # Tesztállomány mérete: 20%
    shuffle=True,               # Keverés: Igen
    random_state=current_year   # Véletlenszám-generátor inicializálása: 2025
)



# Végezzen felügyelt tanítást az alábbi modellekkel és beállításokkal: döntési fa (4 mélység, entrópia homogenitási kritérium),
# logisztikus regresszió (liblinear solverrel) és neurális háló (1 rejtett réteg 4 neuronnal, logisztikus
# aktivációs függvény). A teszt score alapján hasonlítsa össze az illesztett modelleket, melyeket nyomtasson ki. (10 pont)

# 1. Döntési Fa (Decision Tree)
# max_depth=4, criterion='entropy'
print("--- 1. Döntési Fa ---")
dt_model = DecisionTreeClassifier(max_depth=4, criterion='entropy', random_state=2025)
dt_model.fit(X_train, y_train)
dt_score = dt_model.score(X_test, y_test)
print(f"Beállítások: Max. mélység=4, Kritérium=Entrópia")
print(f"Teszt Pontosság (Accuracy): {dt_score:.4f}")

# 2. Logisztikus Regresszió (Logistic Regression)
# solver='liblinear'
print("\n--- 2. Logisztikus Regresszió ---")
# Megjegyzés: A liblinear solver jól működik bináris klasszifikációnál.
# Többosztályos (multiclass) feladatra (mint itt, 10 osztály) a scikit-learn One-vs-Rest (OvR) vagy
# Multinominal módszert használ automatikusan. A liblinear alapértelmezetten OvR-t használ.
lr_model = LogisticRegression(solver='liblinear', random_state=2025, max_iter=1000)
lr_model.fit(X_train, y_train)
lr_score = lr_model.score(X_test, y_test)
print(f"Beállítások: Solver=liblinear")
print(f"Teszt Pontosság (Accuracy): {lr_score:.4f}")

# 3. Neurális Háló (Multi-layer Perceptron - MLP)
# hidden_layer_sizes=(4,), activation='logistic'
print("\n--- 3. Neurális Háló (MLP) ---")
# A MLPClassifier alapértelmezetten One-vs-Rest-et vagy Softmax-ot használ a multiclass feladatokra.
mlp_model = MLPClassifier(hidden_layer_sizes=(4,),
                          activation='logistic',
                          max_iter=1000,
                          random_state=2025)
mlp_model.fit(X_train, y_train)
mlp_score = mlp_model.score(X_test, y_test)
print(f"Beállítások: 1 rejtett réteg (4 neuron), Aktiváció=Logisztikus")
print(f"Teszt Pontosság (Accuracy): {mlp_score:.4f}")



# Számolja ki az 5. pont legjobb modelljére a teszt tévesztési mátrixot. (4 pont)

# Az előrejelzés elkészítése a teszt halmazon
y_pred_logreg = lr_model.predict(X_test)

# --- 2. Konfúziós Mátrix Kiszámítása ---
cm = confusion_matrix(y_test, y_pred_logreg)

# --- 3. Az eredmény vizuális megjelenítése és kiírása ---
print("--- Logisztikus Regresszió Konfúziós Mátrixa ---")
print(cm)
print("-" * 40)



# Ábrázolja a tévesztési mátrixot. (3 pont)

# Kép megjelenítése a ConfusionMatrixDisplay segítségével
fig, ax = plt.subplots(figsize=(8, 8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=digits.target_names)
disp.plot(cmap=plt.cm.Blues, ax=ax)
ax.set_title("Konfúziós Mátrix - Logisztikus Regresszió (Teszt)")
plt.show()



# Végezzen nemfelügyelt tanítást a K-közép módszerrel az input attribútumokon. Határozza meg az optimális klaszterszámot 30-ig a DB indexszel.
# Az optimális klaszterszám mellett vizualizálja a klasztereket egy pontdiagrammon, ahol a két koordináta egy 2 dimenziós PCA eredménye. (13 pont)

# --- 1. DB index kiszámítása K=2-től K=30-ig ---
Max_K = 31  # Maximum klaszterszám: 30
DB = np.zeros((Max_K - 2))  # DB indexek tárolása
K_values = np.arange(2, Max_K)

print("Keresem az optimális K-t...")

for i, K in enumerate(K_values):
    # K-közép futtatása
    kmeans = KMeans(n_clusters=K, random_state=2025, n_init='auto', max_iter=300)
    kmeans.fit(X)

    # DB index kiszámítása
    DB[i] = davies_bouldin_score(X, kmeans.labels_)

# --- 2. DB indexek vizualizálása az optimális K megtalálásához ---
fig = plt.figure(figsize=(10, 5))
plt.title('Davies-Bouldin index a klaszterszám függvényében')
plt.xlabel('Klaszterszám (K)')
plt.ylabel('DB Index (Cél: Minimum)')
plt.plot(K_values, DB, color='blue', marker='o', markersize=5)
plt.xticks(K_values[::2])  # Csak minden második K érték feliratozása
plt.grid(True, linestyle='--')
plt.show()

# --- 3. Optimális K meghatározása (legkisebb DB index) ---
opt_k_index = np.argmin(DB)
opt_k = K_values[opt_k_index]
opt_db_score = DB[opt_k_index]

print("-" * 50)
print(f"Az optimális klaszterszám a legkisebb DB index alapján: K = {opt_k}")
print(f"A hozzá tartozó DB index: {opt_db_score:.4f}")
print("-" * 50)


# --- 1. Adatcsökkentés: PCA 2 komponensre ---
pca = PCA(n_components=2)
X_pc = pca.fit_transform(X) # Az adatok 2 főkomponensre vetítve

# --- 2. Klaszterezés az Optimális K-val ---
kmeans_opt = KMeans(n_clusters=opt_k, random_state=2025, n_init='auto', max_iter=300)
kmeans_opt.fit(X)
opt_labels = kmeans_opt.labels_
opt_centers = kmeans_opt.cluster_centers_

# A klaszter-centroidok levetítése a 2D PCA térbe
centers_pc = pca.transform(opt_centers)

# --- 3. Pontdiagram (Scatter Plot) Létrehozása ---
fig = plt.figure(figsize=(10, 8))
plt.title(f'K-közép Klaszterezés (K={opt_k}) 2D PCA térben')
plt.xlabel('Főkomponens 1 (PC1)')
plt.ylabel('Főkomponens 2 (PC2)')

# Adatpontok ábrázolása a klasztercímkék szerint színezve
plt.scatter(X_pc[:, 0], X_pc[:, 1],
            s=20,
            c=opt_labels,
            cmap='Spectral',
            alpha=0.6,
            label='Adatpontok')

# Centroidok ábrázolása (piros X)
plt.scatter(centers_pc[:, 0], centers_pc[:, 1],
            s=200,
            c='red',
            marker='X',
            edgecolors='black',
            linewidths=1.5,
            label='Centroidok')

# Jelmagyarázat
plt.legend(scatterpoints=1, frameon=True, loc='upper right')
plt.grid(True, linestyle=':')
plt.show()
