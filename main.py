import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, silhouette_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns

# Τίτλος της εφαρμογής
st.title("Ανάπτυξη Εφαρμογής για Ανάλυση Δεδομένων")

# Φόρτωση Δεδομένων
st.header("Φόρτωση Δεδομένων")
uploaded_file = st.file_uploader("Επιλέξτε ένα αρχείο CSV ή Excel", type=["csv", "xlsx"])

if uploaded_file:
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)
    st.write("Προεπισκόπηση Δεδομένων:", data.head())

    # Προδιαγραφές Πίνακα
    st.subheader("Διαστάσεις Πίνακα")
    st.write(f"Αριθμός Δειγμάτων: {data.shape[0]}")
    st.write(f"Αριθμός Χαρακτηριστικών: {data.shape[1] - 1}")

    # Μετατροπή της στήλης ετικετών σε αριθμητικές τιμές
    labels = pd.Categorical(data.iloc[:, -1]).codes

    # Έλεγχος για τουλάχιστον δύο κατηγορίες
    if len(np.unique(labels)) < 2:
        st.error(
            "Τα δεδομένα περιέχουν μόνο μία κατηγορία. Χρειάζονται τουλάχιστον δύο κατηγορίες για την εκπαίδευση των αλγορίθμων.")
    else:
        # 2D Visualization Tab
        st.header("2D Οπτικοποίηση")
        method = st.selectbox("Επιλέξτε Αλγόριθμο Μείωσης Διάστασης", ("PCA", "t-SNE"))
        if method == "PCA":
            pca = PCA(n_components=2)
            components = pca.fit_transform(data.iloc[:, :-1])
        else:
            tsne = TSNE(n_components=2)
            components = tsne.fit_transform(data.iloc[:, :-1])

        plt.figure(figsize=(10, 6))
        plt.scatter(components[:, 0], components[:, 1], c=labels, cmap='viridis')
        plt.xlabel("Συστατικό 1")
        plt.ylabel("Συστατικό 2")
        plt.colorbar()
        st.pyplot(plt)

        # Exploratory Data Analysis (EDA)
        st.subheader("Exploratory Data Analysis (EDA)")
        st.write("Κατανομή Χαρακτηριστικών")
        for column in data.columns[:-1]:
            plt.figure(figsize=(6, 4))
            sns.histplot(data[column], kde=True)
            plt.title(f"Κατανομή του {column}")
            st.pyplot(plt)

        # Tabs Μηχανικής Μάθησης
        st.header("Μηχανική Μάθηση")
        task = st.selectbox("Επιλέξτε Είδος Εργασίας", ("Κατηγοριοποίηση", "Ομαδοποίηση"))

        if task == "Κατηγοριοποίηση":
            st.subheader("Αλγόριθμοι Κατηγοριοποίησης")
            k = st.slider("Επιλέξτε το k για τον k-NN", min_value=1, max_value=20, value=5)
            X = data.iloc[:, :-1]
            y = labels

            # K-Nearest Neighbors
            knn = KNeighborsClassifier(n_neighbors=k)
            knn.fit(X, y)
            y_pred_knn = knn.predict(X)
            accuracy_knn = accuracy_score(y, y_pred_knn)

            # Logistic Regression
            lr = LogisticRegression(max_iter=200)
            lr.fit(X, y)
            y_pred_lr = lr.predict(X)
            accuracy_lr = accuracy_score(y, y_pred_lr)

            st.write(f"Ακρίβεια K-NN: {accuracy_knn:.2f}")
            st.write(f"Ακρίβεια Logistic Regression: {accuracy_lr:.2f}")

            # Εμφάνιση Μετρικών Απόδοσης
            st.write("Μετρικές Απόδοσης K-NN")
            st.text(classification_report(y, y_pred_knn))
            st.write("Μετρικές Απόδοσης Logistic Regression")
            st.text(classification_report(y, y_pred_lr))

            if accuracy_knn > accuracy_lr:
                st.write("Ο K-NN έχει καλύτερη απόδοση.")
            else:
                st.write("Η Logistic Regression έχει καλύτερη απόδοση.")

            # Παρουσίαση Αποτελεσμάτων και Σύγκριση
            st.header("Αποτελέσματα και Σύγκριση")
            st.subheader("K-NN")
            st.write("Ακρίβεια:", accuracy_knn)
            st.write("Μετρικές Απόδοσης:")
            st.text(classification_report(y, y_pred_knn))
            st.subheader("Logistic Regression")
            st.write("Ακρίβεια:", accuracy_lr)
            st.write("Μετρικές Απόδοσης:")
            st.text(classification_report(y, y_pred_lr))

        else:
            st.subheader("Αλγόριθμοι Ομαδοποίησης")
            k = st.slider("Επιλέξτε τον αριθμό των clusters για τον k-Means", min_value=2, max_value=10, value=3)
            X = data.iloc[:, :-1]

            # K-Means
            kmeans = KMeans(n_clusters=k)
            kmeans.fit(X)
            labels_kmeans = kmeans.labels_
            silhouette_kmeans = silhouette_score(X, labels_kmeans)

            # Agglomerative Clustering
            agglomerative = AgglomerativeClustering(n_clusters=k)
            labels_agg = agglomerative.fit_predict(X)
            silhouette_agg = silhouette_score(X, labels_agg)

            st.write(f"Silhouette Score K-Means: {silhouette_kmeans:.2f}")
            st.write(f"Silhouette Score Agglomerative Clustering: {silhouette_agg:.2f}")

            if silhouette_kmeans > silhouette_agg:
                st.write("Ο K-Means έχει καλύτερη απόδοση.")
            else:
                st.write("Η Agglomerative Clustering έχει καλύτερη απόδοση.")

            plt.figure(figsize=(10, 6))
            plt.scatter(components[:, 0], components[:, 1], c=labels_kmeans, cmap='viridis')
            plt.xlabel("Συστατικό 1")
            plt.ylabel("Συστατικό 2")
            plt.colorbar()
            st.pyplot(plt)

            # Παρουσίαση Αποτελεσμάτων και Σύγκριση
            st.header("Αποτελέσματα και Σύγκριση")
            st.subheader("K-Means")
            st.write("Silhouette Score:", silhouette_kmeans)
            st.subheader("Agglomerative Clustering")
            st.write("Silhouette Score:", silhouette_agg)

# Info Tab
st.sidebar.title("Info")
st.sidebar.info("""
    **Ανάπτυξη Εφαρμογής για Ανάλυση Δεδομένων**

    Αυτή η εφαρμογή αναπτύχθηκε για εξόρυξη και ανάλυση δεδομένων με χρήση του Streamlit.

    **Ομάδα Ανάπτυξης:**
    - Θεοφανίδης Συμεών inf2021063 : Ανάπτυξη Κώδικα, Δοκιμές και ανάλυση δεδομένων
    - Σπυρίδων Κίτσος π2018119 : Σχεδίαση Διεπαφής Χρήστη, Τεκμηρίωση και αξιολόγηση αλγορίθμων
    - Στέλιος Βούρλος :

    
""")
