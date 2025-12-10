import numpy as np
from urllib.request import urlopen

from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_curve, auc, RocCurveDisplay,davies_bouldin_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree

# 1. URL cím meghatározása
url = 'https://arato.inf.unideb.hu/ispany.marton/MachineLearning/Datasets/banknote_authentication.txt'

# 2. Adatállomány beolvasása közvetlenül az URL címről
# Mivel a .txt fájl nagy valószínűséggel nem tartalmaz fejlécet, a skiprows=0 beállítással vagy a paraméter elhagyásával olvasunk be.
try:
    raw_data = urlopen(url)
    data = np.loadtxt(raw_data, delimiter=",")
    del raw_data
except Exception as e:
    print(f"Hiba történt az adatállomány beolvasása közben: {e}")
    exit()

# 3. Adatállomány jellemzőinek meghatározása
rekordok_szama = data.shape[0]  # Sorok száma
attrib_szama = data.shape[1] - 1  # Oszlopok száma - 1 (a célváltozó nélkül)

# Célváltozó (osztályok) száma
# A célváltozó az utolsó oszlopban van (index: -1)
y = data[:, -1]
osztalyok_szama = len(np.unique(y)) # Egyedi értékek száma a célváltozóban

# 4. A kért adatok kiírása
print(f"'banknote_authentication.txt' adatállomány jellemzői:")
print(f"Rekordok száma: {rekordok_szama}")
print(f"Attribútumok száma : {attrib_szama}")
print(f"Osztályok száma: {osztalyok_szama}")

# ------------------------------------------------------------------------------------------------------

# --- 2. Attribútumok és célváltozó kinyerése (NumPy indexeléssel) ---

# Adatok kinyerése a tömbből
X_variance = data[:, 0]
X_skewness = data[:, 1]
X_curtosis = data[:, 2]
X_entropy = data[:, 3]
# --- 3. Logikai maszkok létrehozása az osztályok szétválasztására ---

# Hamis (0) bankjegyek (Piros)
mask_class_0 = (y == 0)
# Valódi (1) bankjegyek (Kék)
mask_class_1 = (y == 1)

# --- 4. Pontdiagram készítése ---
fig = plt.figure(figsize=(8, 6))
plt.title('Variance + Entropy')
plt.xlabel('Variance')
plt.ylabel('Entropy')

# 0. osztály (Hamis) ábrázolása: A maszk True értékeinél kiválasztjuk az adatokat
plt.scatter(X_variance[mask_class_0], X_entropy[mask_class_0],
            c='red',
            s=20,
            label='Hamis (0)',
            alpha=0.6)

# 1. osztály (Valódi) ábrázolása:
plt.scatter(X_variance[mask_class_1], X_entropy[mask_class_1],
            c='blue',
            s=20,
            label='Valódi (1)',
            alpha=0.6)

# Legend (Jelmagyarázat)
plt.legend()
# plt.grid(True, linestyle=':')
plt.show()

# -----------------------------------------------------------------------------------------

# Jellemzők (X) és Célváltozó (y) szétválasztása
x = data[:, :-1]  # Minden sor, az utolsó oszlop kivételével
y = data[:, -1]   # Minden sor, az utolsó oszlop (Target)

# --- Particionálás a megadott beállításokkal ---
# Teszt méret: 20% (test_size=0.20)
# Keverés: Igen (shuffle=True)
# Véletlenszám-generátor inicializálása: 100 (random_state=100)


x_train, x_test, y_train, y_test = train_test_split(
    x,
    y,
    test_size=0.20,             # Tesztállomány mérete: 20%
    shuffle=True,               # Keverés: Igen
    random_state=100  # Véletlenszám-generátor inicializálása: 100
)


# -----------------------------------------------------------------------------------------
# Modellek Betanítása és Kiértékelése
# -----------------------------------------------------------------------------------------


print("\n1. Döntési Fa")
dt_model = DecisionTreeClassifier(max_depth=5, criterion='entropy')
dt_model.fit(x_train, y_train)
dt_score = dt_model.score(x_test, y_test)
print(f"Teszt Pontosság: {dt_score:.4f}")

print("2. Logisztikus Regresszió")
lr_model = LogisticRegression(solver='liblinear')
lr_model.fit(x_train, y_train)
lr_score = lr_model.score(x_test, y_test)
print(f"Teszt Pontosság: {lr_score:.4f}")

print("3. Neurális Háló")
mlp_model = MLPClassifier(hidden_layer_sizes=(2,),
                          activation='logistic',
                          solver='lbfgs',
                          max_iter=5000)
mlp_model.fit(x_train, y_train)
mlp_score = mlp_model.score(x_test, y_test)
print(f"Teszt Pontosság: {mlp_score:.4f}")

# -----------------------------------------------------------------------------------------

# --- 1. Döntési Fa Ábrázolása ---

# Az attribútumok nevei a vizualizációhoz
feature_names = ['variance', 'skewness', 'curtosis', 'entropy']
class_names = ['Hamis (0)', 'Valódi (1)']

fig, ax = plt.subplots(figsize=(20, 10))
plot_tree(dt_model,
          feature_names=feature_names,
          class_names=class_names,
          filled=True,
          ax=ax,
          fontsize=8)
plt.title("Döntési Fa", fontsize=16)
plt.show()


# --- 2. Tévesztési Mátrix (Konfúziós Mátrix) Kiszámítása és Rajzolása ---

y_pred_dt = dt_model.predict(x_test)
cm = confusion_matrix(y_test, y_pred_dt)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=class_names)
disp.plot(cmap=plt.cm.Greens)
plt.show()

# ---------------------------------------------------------------------------------------------

# --- 1. Ábra inicializálása ---
fig, ax = plt.subplots(figsize=(8, 8))
lw = 2 # Vonalvastagság
dt_proba = dt_model.predict_proba(x_test)[:, 1]
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_proba)
auc_dt = auc(fpr_dt, tpr_dt)
print(f"Döntési Fa: AUC = {auc_dt:.4f}")

dt_display = RocCurveDisplay(fpr=fpr_dt, tpr=tpr_dt, roc_auc=auc_dt,
                             estimator_name=f'Döntési Fa (AUC = {auc_dt:.4f})')
dt_display.plot(ax=ax, color='red')


lr_proba = lr_model.predict_proba(x_test)[:, 1]
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
auc_lr = auc(fpr_lr, tpr_lr)
print(f"Logisztikus Regresszió: AUC = {auc_lr:.4f}")
lr_display = RocCurveDisplay(fpr=fpr_lr, tpr=tpr_lr, roc_auc=auc_lr,
                             estimator_name=f'Logisztikus Regresszió (AUC = {auc_lr:.4f})')
lr_display.plot(ax=ax, color='blue')

mlp_proba = mlp_model.predict_proba(x_test)[:, 1]
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, mlp_proba)
auc_mlp = auc(fpr_mlp, tpr_mlp)
print(f"Neurális Háló: AUC = {auc_mlp:.4f}")
mlp_display = RocCurveDisplay(fpr=fpr_mlp, tpr=tpr_mlp, roc_auc=auc_mlp,
                              estimator_name=f'Neurális Háló (AUC = {auc_mlp:.4f})')
mlp_display.plot(ax=ax, color='green')


# --- A véletlenszerű tipp (Random Guess) vonalának rajzolása ---
ax.plot([0, 1], [0, 1], color='black', lw=lw, linestyle='--', label='Véletlenszerű Tipp (AUC=0.50)')

# --- Diagram beállítások ---
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Hamis Pozitív Ráta (False Positive Rate)')
ax.set_ylabel('Valódi Pozitív Ráta (True Positive Rate)')
ax.set_title('Osztályozók ROC Görbéjének Összehasonlítása (Bankjegy Adatok)')
ax.legend(loc="lower right")
plt.show()

# ----------------------------------------------------------------------------------------------------

DB = []
K_values = range(2, 12)

print(f"Keresés a tartományban: K = {min(K_values)} - {max(K_values)}")

for K in K_values:
    kmeans_temp = KMeans(n_clusters=K, random_state=100, n_init='auto')
    kmeans_temp.fit(x)
    labels_temp = kmeans_temp.labels_
    db_score = davies_bouldin_score(x, labels_temp)
    DB.append(db_score)
    # print(f"K={K}: DB Index = {db_score:.4f}")

# DB indexek vizualizációja görbén
plt.figure(figsize=(8, 5))
plt.title('Davies-Bouldin index görbe')
plt.xlabel('Klaszterszám (K)')
plt.ylabel('DB index')
plt.plot(K_values, DB, color='blue', marker='o', markersize=8)
plt.xticks(K_values)
plt.show()

# Optimális K kiválasztása (ahol a DB index a legkisebb)
min_db_index = np.argmin(DB)
optimal_K = K_values[min_db_index]
print(f"\nAz optimális klaszterszám: K = {optimal_K}")
print(f"Legkisebb DB érték: {DB[min_db_index]:.4f}")

# ---------------------------------------------------------------------------------------------

# 2. Klaszterezés az OPTIMÁLIS K értékkel
kmeans_opt = KMeans(n_clusters=optimal_K, random_state=100, n_init='auto')
kmeans_opt.fit(x)
labels_opt = kmeans_opt.labels_
centers_opt = kmeans_opt.cluster_centers_

# 3. PCA dimenziócsökkentés a vizualizációhoz (2 komponens)
pca = PCA(n_components=2)
pca.fit(x)
x_pc = pca.transform(x)          # Adatpontok a PCA térben
centers_pc = pca.transform(centers_opt) # Centroidok a PCA térben

# 4. Vizualizáció a PCA térben
plt.figure(figsize=(10, 8))
plt.title(f'Klaszterek vizualizációja PCA térben (K={optimal_K})')
plt.xlabel('PC1')
plt.ylabel('PC2')

# Adatpontok kirajzolása (színezés a klasztercímkék szerint)
# A 'viridis' vagy 'tab10' colormap segít elkülöníteni a klasztereket
plt.scatter(x_pc[:, 0], x_pc[:, 1], c=labels_opt, cmap='viridis', s=50, alpha=0.6, label='Adatpontok')

# Centroidok kirajzolása
plt.scatter(centers_pc[:, 0], centers_pc[:, 1], c='red', s=200, marker='X', label='Kozeppontok')

plt.legend()
plt.grid(True, linestyle=':')
plt.show()




