import pandas as pd
import ast
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import ComplementNB
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, make_scorer, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, BaggingClassifier, StackingClassifier
from sklearn.preprocessing import MaxAbsScaler, MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns

# CSV-Dateien einlesen
df = pd.read_csv('df_kommersbuch.csv', delimiter=';')
df_tfidf = pd.read_csv('tfidf_kommersbuch.csv', sep=';')
df_bow = pd.read_csv('bow_kommersbuch.csv', sep=';')
df_ig = pd.read_csv('ig_kommersbuch.csv', sep=';')

# Filterung von TF-IDF und Bag of Words nach dem Informationsgewinn
# Liste der relevanten Wörter aus df_ig
relevant_words = df_ig['Wort'].unique()
# Zusätzliche Spalten um sicher zu gehen
extra_columns = ['Label', 'Titel']
# Filtern der Spalten in df_tfidf und df_bow
filtered_tfidf = df_tfidf[extra_columns + [col for col in df_tfidf.columns if col in relevant_words]]
filtered_bow = df_bow[extra_columns + [col for col in df_bow.columns if col in relevant_words]]


# Interpunktionsfeatures werden in ein geeignetes Format umgewandelt, im DF mit fester Reihenfolge der ersten Elemente in jedem Tupel dargestellt mit der gleichen Anzahl von den Tupeln in jeder Zeile
def parse_punctuation_features(punctuation_str):
    # String in Liste von Tupeln konvertieren
    features = ast.literal_eval(punctuation_str)  
    # Nur die Zahlenwerte extrahieren
    return [count for _, count in features]

# Funktion zur Umwandlung der LDA-Topics in eine Liste von Gewichtungen, dieselben Bedingungen wie bei der Funktion oben
def parse_lda_features(lda_str):
    features = ast.literal_eval(lda_str)
    return [weight for _, weight in features]

# POS-Tags umgewandelt in Vektoren, da im DF ungeordnet dargestellt
def extract_pos_vectors(pos_str, tag_to_index):
    vector = np.zeros(len(tag_to_index))
    pos_list = ast.literal_eval(pos_str)
    for tag, count in pos_list:
        if tag in tag_to_index:
            vector[tag_to_index[tag]] = count
    return vector

# Abhängigkeiten umgewandelt in Vektoren, da im DF ungeordnet dargestellt
def extract_dependency_vectors(dep_str, dep_to_index):
    vector = np.zeros(len(dep_to_index))
    dep_list = ast.literal_eval(dep_str)
    for dep, count in dep_list:
        if dep in dep_to_index:
            vector[dep_to_index[dep]] = count
    return vector

# Named Entities umgewandelt in Vektoren, da im DF ungeordnet dargestellt
def extract_ner_vectors(ner_str, ner_to_index):
    vector = np.zeros(len(ner_to_index))
    ner_list = ast.literal_eval(ner_str)
    for (entity, label), count in ner_list:
        key = (entity, label)
        if key in ner_to_index:
            vector[ner_to_index[key]] = count
    return vector

# Emolex-Entities umgewandelt in Vektoren, da im DF ungeordnet dargestellt
def extract_emotion_vectors(emotion_str, emotion_to_index):
    vector = np.zeros(len(emotion_to_index))
    try:
        emotion_list = ast.literal_eval(emotion_str)
    except (ValueError, SyntaxError) as e:
        print(f"Fehler beim Parsen von Emotion-Features: {emotion_str}. Fehler: {e}")
        return vector  # Rückgabe des Nullvektors bei Fehlern
    for emotion, count in emotion_list:
        if emotion in emotion_to_index:
            vector[emotion_to_index[emotion]] = count 
    return vector

# Alle POS-Tags sammeln und zu Indexen zuordnen
all_pos_tags = set(tag for pos_list in df['POS_Tags'] 
                   for tag, _ in ast.literal_eval(pos_list))
tag_to_index = {tag: idx for idx, tag in enumerate(all_pos_tags)}
# Alle Abhängigkeiten
all_dependencies = set(dep for dep_list in df['Dependency_Features'] 
                        for dep, _ in ast.literal_eval(dep_list))
dep_to_index = {dep: idx for idx, dep in enumerate(all_dependencies)}
# Alle NER
all_entities = set((entity, label) for ner_list in df['NER_Entities'] 
                    for (entity, label), _ in ast.literal_eval(ner_list))
ner_to_index = {entity: idx for idx, entity in enumerate(all_entities)}
# Alle Emolex
all_emotions = set(emotion for emot_list in df['Emotion_Features'] 
                   for emotion, _ in ast.literal_eval(emot_list))
emotion_to_index = {emotion: idx for idx, emotion in enumerate(all_emotions)}

''' Features '''

# Labels für die Klassifikation, die werden in allen Klassifizierungen benutzt
y = df['Label']

# Interpunktions-Features, Format: Liste mit Frequenzen, da die Reihenfolge von den Interpunktionszeichen im DF fest ist
X_punkt = df['Punctuation_Features'].apply(parse_punctuation_features).tolist()

# POS-Features
X_pos = np.array([extract_pos_vectors(pos, tag_to_index) for pos in df['POS_Tags']])

# Dependency-Features
X_dep = np.array([extract_dependency_vectors(dep, dep_to_index) for dep in df['Dependency_Features']])

# LDA-Features
X_lda = np.array([parse_lda_features(lda) for lda in df['LDA_Topics']]) # bei LDA ist die F1 oft niedrig (14-15) und ist nah an die zufällige Wahrscheinlichkeit, daher ist diese Feature nutzlos und zieht den Klassifikator runter, was aber zeigt, dass Lieder ähnlicher sind, als auf den ersten Blick erscheint

# NER-Feature
X_ner = np.array([extract_ner_vectors(ner, ner_to_index) for ner in df['NER_Entities']])

# Emolex-Features
X_emot = np.array([extract_emotion_vectors(emot, emotion_to_index) for emot in df['Emotion_Features']])

# Skalieren mit MinMaxScaler (zwischen 0 und 1), da es bei BoW und TF-IDF keine negativen Werte gibt
scaler = MinMaxScaler()
# TF-IDF
X_tfidf = pd.DataFrame(scaler.fit_transform(df_tfidf.drop(columns=['Label', 'Titel'])), columns=df_tfidf.drop(columns=['Label', 'Titel']).columns)
X_tfidf = X_tfidf.to_numpy()

# Bag of Words
X_bow = pd.DataFrame(scaler.fit_transform(df_bow.drop(columns=['Label', 'Titel'])), columns=df_bow.drop(columns=['Label', 'Titel']).columns)
X_bow = X_bow.to_numpy()

# TF-IDF und BoW mit Informationsgewinn
X_filtered_tfidf = filtered_tfidf.drop(columns=['Label', 'Titel']).to_numpy()
X_filtered_tfidf = scaler.fit_transform(X_filtered_tfidf)
X_filtered_bow = filtered_bow.drop(columns=['Label', 'Titel']).to_numpy()
X_filtered_bow = scaler.fit_transform(X_filtered_bow)

# Kombination aus POS- und Interpunktions-Features
#X_punkt_pos_combined = [np.concatenate((pos, punct)) for pos, punct in zip(X_pos, X_punkt)]
# Kombination aus POS- und Abhängigkeits-Features
#X_pos_dep_combined = [np.concatenate((pos, dep)) for pos, dep in zip(X_pos, X_punkt)]
# Kombination aus Interpunktions- und Abhängigkeits-Features
#X_punkt_dep_combined = [np.concatenate((punkt, dep)) for punkt, dep in zip(X_pos, X_punkt)]
# war bei F1 0.28 ± 0.05, nicht effizient
# Kombination aus Interpunktions-, POS- und Dependency-Features
#X_dep_punkt_pos_combined = [np.concatenate((pos, punct, dep)) for pos, punct, dep in zip(X_pos, X_punkt, X_dep)]
# Kombination aus Interpunktions-, POS-, Dependency-Features und LDA
#X_dep_punkt_pos_lda_combined = [np.concatenate((pos, punct, dep, lda)) for pos, punct, dep, lda in zip(X_pos, X_punkt, X_dep, X_ner)]
# Kombination von Features: es wurden verschiedene Kombinationen ausprobiert und eine Analyse den effizientesten Features durchgfeührt, hier ist das effizienteste Ergebnis 
# AM BESTEN: 
X_combined = np.array([np.concatenate((tfidf, bow)) for tfidf, bow in zip( X_tfidf, X_bow)])
#X_combined = np.array([np.concatenate((punct, pos, dep, lda, ner, emot, tfidf, bow, ftfidf, fbow)) for punct, pos, dep, lda, ner, emot, tfidf, bow, ftfidf, fbow in zip(X_punkt, X_pos, X_dep, X_lda, X_ner, X_emot, X_tfidf, X_bow, X_filtered_tfidf, X_filtered_bow)])

# Train-Test-Split mit Skalierung mittels MaxAbsScaler (zwischen -1 und 1, 0-Werte bleiben unverändert)
def split_data(X, y, test_size=0.2, random_state=42):
    # Train-Test-Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    # MaxAbsScaler
    scaler = MaxAbsScaler()
    # Scaler nur auf Trainingsdaten anwenden
    X_train_scaled = scaler.fit_transform(X_train)
    # Testdaten transformieren (mit denselben Parametern)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, y_train, y_test

''' Train-Test-Splits für die verschiedenen Feature-Sets '''
X_train_punkt, X_test_punkt, y_train_punkt, y_test_punkt = split_data(X_punkt, y, test_size=0.2, random_state=42)
X_train_pos, X_test_pos, y_train_pos, y_test_pos = split_data(X_pos, y, test_size=0.2, random_state=42)
X_train_dep, X_test_dep, y_train_dep, y_test_dep = split_data(X_dep, y, test_size=0.2, random_state=42)
X_train_lda, X_test_lda, y_train_lda, y_test_lda = split_data(X_lda, y, test_size=0.2, random_state=42)
X_train_ner, X_test_ner, y_train_ner, y_test_ner = split_data(X_ner, y, test_size=0.2, random_state=42)
X_train_emot, X_test_emot, y_train_emot, y_test_emot = split_data(X_emot, y, random_state=42)
X_train_tfidf, X_test_tfidf, y_train_tfidf, y_test_tfidf = train_test_split(X_tfidf, y, test_size=0.3, random_state=42)
X_train_bow, X_test_bow, y_train_bow, y_test_bow = train_test_split(X_bow, y, test_size=0.3, random_state=42)
X_train_ftfidf, X_test_ftfidf, y_train_ftfidf, y_test_ftfidf = train_test_split(X_filtered_tfidf, y, test_size=0.3, random_state=42)
X_train_fbow, X_test_fbow, y_train_fbow, y_test_fbow= train_test_split(X_filtered_bow, y, test_size=0.3, random_state=42)

# Kombinierte Features
X_train_combined, X_test_combined, y_train_combined, y_test_combined = split_data(X_combined, y, test_size=0.3, random_state=42)

''' Random Forest Klassifikator, alle Angaben sind nach dem Tuning (s. unten) erfolgt '''
rf_punkt = RandomForestClassifier(n_estimators=100, class_weight='balanced', min_samples_split=5, min_samples_leaf=1, max_depth=10, criterion='gini', random_state=42)
rf_pos = RandomForestClassifier(n_estimators=100, class_weight='balanced', min_samples_split=5, min_samples_leaf=1,criterion='gini', random_state=42)
rf_dep = RandomForestClassifier(n_estimators=300, class_weight='balanced', min_samples_split=2, min_samples_leaf=2, criterion='entropy', max_depth=10, random_state=42)
rf_lda = RandomForestClassifier(n_estimators=200, class_weight='balanced', min_samples_split=3,  min_samples_leaf=2,criterion='gini', random_state=42)
rf_ner = RandomForestClassifier(n_estimators=400, class_weight='balanced', min_samples_split=5, min_samples_leaf=1, max_depth=40, criterion='gini', random_state=42)
rf_emot = RandomForestClassifier(n_estimators=200, min_samples_split=5, criterion='gini', min_samples_leaf=2, max_depth=10, random_state=42)  
rf_tfidf = RandomForestClassifier(n_estimators=200, criterion='gini', min_samples_leaf=1, min_samples_split=5, max_depth=20, random_state=42)
rf_bow = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_leaf=2, min_samples_split=2, random_state=42)
rf_ig_tfidf = RandomForestClassifier(n_estimators=300, criterion='gini', min_samples_leaf=1, min_samples_split=3, random_state=42) 
rf_ig_bow = RandomForestClassifier(n_estimators=100, criterion='gini', min_samples_leaf=3, max_depth=10, min_samples_split=2, random_state=42) 
rf_combined = RandomForestClassifier(n_estimators=400, criterion='gini', min_samples_leaf=2, min_samples_split=5, max_depth=20, random_state=42)

''' Naive Bayes Klassifikator, alle Angaben sind nach dem Tuning (s. unten) erfolgt '''
# Interpunktions-Features
nb_punkt = ComplementNB(alpha=2.0, fit_prior=True, force_alpha=True)
# POS-Tags
nb_pos = ComplementNB(alpha=0.01, fit_prior=True, force_alpha=True)
# Dependency-Features
nb_dep = ComplementNB(alpha=5.0, fit_prior=True, force_alpha=True)
# LDA-Features
nb_lda = ComplementNB(alpha=5.0, fit_prior=True, force_alpha=True)
# NER-Features
nb_ner = ComplementNB(alpha=5.0, fit_prior=True, force_alpha=True)
# Emolex-Emotion-Features
nb_emot = ComplementNB(alpha=10.0, fit_prior=True, force_alpha=True)
# TF-IDF
nb_tfidf = ComplementNB(fit_prior=True, norm=True)
# Bag-of-Words
nb_bow = ComplementNB(fit_prior=True, norm=True)
# IG-gefiltertes TF-IDF
nb_ig_tfidf = ComplementNB(alpha=0.1, fit_prior=True, force_alpha=True, norm=True)
# IG-gefiltertes BoW
nb_ig_bow = ComplementNB(alpha=0.1, fit_prior=True, force_alpha=True, norm=True)

# Kombination aller Features
nb_combined = ComplementNB(alpha=1.0, fit_prior=True, force_alpha=True, norm=True)

''' Klassifikation mit Logistischer Regression, alle Angaben sind nach dem Tuning (s. unten) erfolgt, es war nicht mehr sinnvoll, den  multi_class zu implementieren, da dieser Parameter ab der Version 1.7 entfernt wird, in der Warnung ist aber ein Hinweis auf OneVsRestClassifier enthalten. Auch wurde gewarnt, dass n_jobs=-1 bei solver='liblinear' keinen Effekt hat, weshalb dieser Parameter bei dem entsprechenden Solver gelöscht wure'''
# Interpunktions-Features
lr_punkt = LogisticRegression(C=10.0, max_iter=500, penalty='l1', solver='saga', tol=0.001, class_weight='balanced', random_state=42, n_jobs=-1)

# POS-Tags
lr_pos = LogisticRegression(C=10.0, max_iter=500, penalty='l1', solver='liblinear', tol=0.0001, class_weight='balanced', random_state=42)

# Dependency-Features
lr_dep = LogisticRegression(C=10.0, max_iter=5000, penalty='l1', solver='saga', tol=0.0001, class_weight='balanced', random_state=42, n_jobs=-1)

# LDA-Features
lr_lda = LogisticRegression(C=1.0, max_iter=500, penalty='l2', solver='lbfgs', tol=0.001, class_weight='balanced', random_state=42, n_jobs=-1)

# NER-Features
lr_ner = LogisticRegression(C=10.0, max_iter=5000, penalty='l2', solver='saga', tol=0.0001, class_weight='balanced', random_state=42, n_jobs=-1)

# Emolex-Emotion-Features
lr_emot = LogisticRegression(C=10.0, max_iter=500, penalty='l1', solver='saga', tol=0.001, class_weight='balanced', random_state=42, n_jobs=-1)

# TF-IDF
lr_tfidf = LogisticRegression(C=10.0, max_iter=5000, penalty='l2', solver='saga', tol=0.0001, class_weight='balanced', random_state=42, n_jobs=-1)

# Bag-of-Words
lr_bow = LogisticRegression(C=10.0, max_iter=5000, penalty='l1', solver='saga', tol=0.0001, class_weight='balanced', random_state=42, n_jobs=-1)

# IG-gefiltertes TF-IDF
lr_ig_tfidf = LogisticRegression(C=10.0, max_iter=5000, penalty='l1', solver='liblinear', tol=0.01, class_weight='balanced', random_state=42)

# IG-gefiltertes BoW
lr_ig_bow = LogisticRegression(C=10.0, max_iter=5000, penalty='l1', solver='saga', tol=0.0001, class_weight='balanced', random_state=42, n_jobs=-1)

# Kombination aller Features
lr_combined = LogisticRegression(C=10.0, max_iter=5000, penalty='l2', solver='lbfgs', tol=0.0001, class_weight='balanced', random_state=42, n_jobs=-1)

''' Support Vector Machines, alle Angaben sind nach dem Tuning (s. unten) erfolgt '''
# Interpunktions-Features
svm_punkt = SVC(C=10.0, decision_function_shape='ovo', gamma='scale', kernel='rbf', max_iter=1000, probability=True, shrinking=True, tol=0.001, class_weight='balanced', random_state=42)
# POS-Tags
svm_pos = SVC(C=100.0, decision_function_shape='ovo', gamma='scale', kernel='linear', max_iter=-1, probability=True, shrinking=False, tol=0.001, class_weight='balanced', random_state=42)
# Dependency-Features
svm_dep = SVC(C=10.0, decision_function_shape='ovo', gamma='scale', kernel='rbf', max_iter=1000, probability=True, shrinking=True, tol=0.001, class_weight='balanced', random_state=42)
# LDA-Features
svm_lda = SVC(C=100.0, decision_function_shape='ovo', gamma='scale', kernel='rbf', max_iter=-1, probability=True, shrinking=True, tol=0.001, class_weight='balanced', random_state=42)
# NER-Features
svm_ner = SVC(C=100.0, decision_function_shape='ovo', gamma=0.01, kernel='rbf', max_iter=-1, probability=True, shrinking=True, tol=0.001, class_weight='balanced', random_state=42)
# Emolex-Emotion-Features
svm_emot = SVC(C=100.0, decision_function_shape='ovo', gamma=0.01, kernel='rbf', max_iter=-1, probability=True, shrinking=True, tol=0.0001, class_weight='balanced', random_state=42)
# TF-IDF
svm_tfidf = SVC(C=1.0, decision_function_shape='ovo', gamma='scale', kernel='linear', max_iter=1000, probability=True, shrinking=True, tol=0.001,class_weight='balanced', random_state=42)
# Bag-of-Words
svm_bow = SVC(C=100.0, decision_function_shape='ovo', gamma='scale', kernel='linear', max_iter=-1, probability=True, shrinking=True, tol=0.0001, class_weight='balanced', random_state=42)
# IG-gefiltertes TF-IDF
svm_ig_tfidf = SVC(C=100.0, decision_function_shape='ovo', gamma='scale', kernel='linear', max_iter=-1, probability=True, shrinking=True, tol=0.001, class_weight='balanced', random_state=42)
# IG-gefiltertes BoW
svm_ig_bow = SVC(C=100.0, decision_function_shape='ovo', gamma='auto', kernel='rbf', max_iter=-1, probability=True, shrinking=True, tol=0.001, class_weight='balanced', random_state=42)

# Kombination aller Features
svm_combined = SVC(C=1.0, decision_function_shape='ovo', gamma='scale', kernel='linear', max_iter=-1, probability=True, shrinking=True, tol=0.001, class_weight='balanced', random_state=42)

# Liste mit RF Klassifikatoren
rf_classifiers = [
    #('punkt', rf_punkt), 
    #('pos', rf_pos),
    #('dep', rf_dep),
    #('ner', rf_ner),
    #('emot', rf_emot),
    ('tfidf', rf_tfidf),
    ('bow', rf_bow)
]

# Liste mit NB Klassifikatoren
nb_classifiers = [
    ('punkt', nb_punkt), 
    ('pos', nb_pos),
    ('dep', nb_dep),
    ('ner', nb_ner),
    ('emot', nb_emot),
    ('tfidf', nb_tfidf),
    ('bow', nb_bow)
]

# Liste mit LR Klassifikatoren
lr_classifiers = [
    ('punkt', lr_punkt), 
    ('pos', lr_pos),
    ('dep', lr_dep),
    #('lda', lr_lda),
    ('ner', lr_ner),
    ('emot', lr_emot),
    ('tfidf', lr_tfidf),
    ('bow', lr_bow)
]

# Liste mit SVM Klassifikatoren
svm_classifiers = [
    #('punkt', lr_punkt), 
    #('pos', lr_pos),
    #('dep', lr_dep),
    #('lda', lr_lda),
    #('ner', lr_ner),
    #('emot', lr_emot),
    ('tfidf', lr_tfidf),
    ('bow', lr_bow)
]

# Ensemble-Klassifikator mit VotingClassifier mit NB
voting_ensemble = VotingClassifier(estimators=nb_classifiers, voting='soft', weights=[1, 2, 2, 1, 1, 5, 5], n_jobs=-1) # Als estimators wurden rf_classifiers, nb_classifiers, lr_classifiers und svm_classifiers getestet

base_classifier = ComplementNB(alpha=1.0, fit_prior=True, force_alpha=True, norm=True)
# Ensemble-Klassifikator mit BaggingClassifier mit NB
bagging_ensemble = BaggingClassifier( # es wurden verschiedene Basisklassifikatoren ausprobiert
    estimator=base_classifier,
    n_estimators=150, #war 10, 50, 100
    max_samples=1.0,  # Anteil der Daten pro Stichprobe, war 0.8
    max_features=1.0,  # Anteil der Features pro Stichprobe, was 1.0, 0.5
    random_state=42,
    bootstrap=False,
    bootstrap_features=False,
    warm_start=True,
    #oob_score=True, #warm_start ist nur ohne oob_score verfügbar und umgekehrt
    n_jobs=-1  # Parallelisierung
)

# Ensemble-Klassifikator mit StackingClassifier mit SVM
stacking_ensemble = StackingClassifier(
    estimators=svm_classifiers, # Als estimators wurden rf_classifiers, nb_classifiers, lr_classifiers und svm_classifiers getestet
    final_estimator=ComplementNB(alpha=1.0, fit_prior=True, force_alpha=True, norm=True),
    cv=12,  # Kreuzvalidierung für das Lernen
    stack_method='predict_proba',
    passthrough=True,
    n_jobs=-1
)

# Training der Modelle für RF
rf_punkt.fit(X_train_punkt, y_train_punkt)
rf_pos.fit(X_train_pos, y_train_pos)
rf_dep.fit(X_train_dep, y_train_dep)
rf_lda.fit(X_train_lda, y_train_lda)
rf_ner.fit(X_train_ner, y_train_ner)
rf_emot.fit(X_train_emot, y_train_emot)
rf_tfidf.fit(X_train_tfidf, y_train_tfidf)
rf_bow.fit(X_train_bow, y_train_bow)
rf_ig_tfidf.fit(X_train_ftfidf, y_train_ftfidf)
rf_ig_bow.fit(X_train_fbow, y_train_fbow)

rf_combined.fit(X_train_combined, y_train_combined)

# Training der Modelle für NB
nb_punkt.fit(X_train_punkt, y_train_punkt)
nb_pos.fit(X_train_pos, y_train_pos)
nb_dep.fit(X_train_dep, y_train_dep)
nb_lda.fit(X_train_lda, y_train_lda)
nb_ner.fit(X_train_ner, y_train_ner)
nb_emot.fit(X_train_emot, y_train_emot)
nb_tfidf.fit(X_train_tfidf, y_train_tfidf)
nb_bow.fit(X_train_bow, y_train_bow)
nb_ig_tfidf.fit(X_train_ftfidf, y_train_ftfidf)
nb_ig_bow.fit(X_train_fbow, y_train_fbow)

nb_combined.fit(X_train_combined, y_train_combined)

# Training der Logistischen Regression
lr_punkt.fit(X_train_punkt, y_train_punkt)
lr_pos.fit(X_train_pos, y_train_pos)
lr_dep.fit(X_train_dep, y_train_dep)
lr_lda.fit(X_train_lda, y_train_lda)
lr_ner.fit(X_train_ner, y_train_ner)
lr_emot.fit(X_train_emot, y_train_emot)
lr_tfidf.fit(X_train_tfidf, y_train_tfidf)
lr_bow.fit(X_train_bow, y_train_bow)
lr_ig_tfidf.fit(X_train_ftfidf, y_train_ftfidf)
lr_ig_bow.fit(X_train_fbow, y_train_fbow)

lr_combined.fit(X_train_combined, y_train_combined)

# Training der Support Vector Machines
svm_punkt.fit(X_train_punkt, y_train_punkt)
svm_pos.fit(X_train_pos, y_train_pos)
svm_dep.fit(X_train_dep, y_train_dep)
svm_lda.fit(X_train_lda, y_train_lda)
svm_ner.fit(X_train_ner, y_train_ner)
svm_emot.fit(X_train_emot, y_train_emot)
svm_tfidf.fit(X_train_tfidf, y_train_tfidf)
svm_bow.fit(X_train_bow, y_train_bow)
svm_ig_tfidf.fit(X_train_ftfidf, y_train_ftfidf)
svm_ig_bow.fit(X_train_fbow, y_train_fbow)

svm_combined.fit(X_train_combined, y_train_combined)

# Training der Ensemble-Modelle
voting_ensemble.fit(X_train_combined, y_train_combined)
bagging_ensemble.fit(X_train_combined, y_train_combined)
bagging_ensemble.n_estimators = 300     # Erhöhung von n_estimator für warm_start
bagging_ensemble.fit(X_train_combined, y_train_combined)
stacking_ensemble.fit(X_train_combined, y_train_combined)

# 10-Fold-Kreuzvalidierung und Bewertung
def evaluate_model(model, X, y, feature_name="Features", n_splits=5):
    # StratifiedKFold, da die Kategorien leicht unbalanciert sind
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    # Kreuzvalidierung für Genauigkeit
    accuracy_scores = cross_val_score(model, X, y, cv=skf, scoring='accuracy')

    # Kreuzvalidierung für F1-Score
    f1_scores = cross_val_score(model, X, y, cv=skf, scoring='f1_weighted')

    print(f'Kreuzvalidierungsergebnisse für {feature_name}:')
    print(f'  Durchschnittliche Genauigkeit: {accuracy_scores.mean():.2f} ± {accuracy_scores.std():.2f}')
    print(f'  Durchschnittlicher F1-Score: {f1_scores.mean():.2f} ± {f1_scores.std():.2f}')
    print('-' * 50)

# Anwendung der Funktion auf die Modelle
# Random Forest
print("RANDOM FOREST")
print("Bewertung für Punktuations-Features:")
evaluate_model(rf_punkt, X_test_punkt, y_test_punkt, feature_name="Punctuation_Features")

print("Bewertung für POS-Features:")
evaluate_model(rf_pos, X_test_pos, y_test_pos, feature_name="POS_Features")

print("Bewertung für Dependency-Features:")
evaluate_model(rf_dep, X_test_dep, y_test_dep, feature_name="Dependency_Features")

print("Bewertung für LDA-Features:")
evaluate_model(rf_lda, X_test_lda, y_test_lda, feature_name="LDA_Entities")

print("Bewertung für NER-Features:")
evaluate_model(rf_ner, X_test_ner, y_test_ner, feature_name="NER_Entities")

print("Bewertung für Emolex-Features:")
evaluate_model(rf_emot, X_test_emot, y_test_emot, feature_name="Emolex")

print("Bewertung für TF-IDF:")
evaluate_model(rf_tfidf, X_test_tfidf, y_test_tfidf, feature_name="TF-IDF")

print("Bewertung für Bag of Words:")
evaluate_model(rf_bow, X_test_bow, y_test_bow, feature_name="BoW")

print("Bewertung für BoW und TF-IDF mit IG:")
evaluate_model(rf_ig_tfidf, X_test_ftfidf, y_test_ftfidf, feature_name="IG+TF-IDF")
evaluate_model(rf_ig_bow, X_test_fbow, y_test_fbow, feature_name="IG+BoW")

# Naive Bayes
print("NAIVE BAYES")
print("Bewertung für Punktuations-Features:")
evaluate_model(nb_punkt, X_test_punkt, y_test_punkt, feature_name="Punctuation_Features")

print("Bewertung für POS-Features:")
evaluate_model(nb_pos, X_test_pos, y_test_pos, feature_name="POS_Features")

print("Bewertung für Dependency-Features:")
evaluate_model(nb_dep, X_test_dep, y_test_dep, feature_name="Dependency_Features")

print("Bewertung für LDA:")
evaluate_model(nb_lda, X_test_lda, y_test_lda, feature_name="LDA")

print("Bewertung für NER-Features:")
evaluate_model(nb_ner, X_test_ner, y_test_ner, feature_name="NER_Entities")

print("Bewertung für Emolex-Features:")
evaluate_model(nb_emot, X_test_emot, y_test_emot, feature_name="Emolex")

print("Bewertung für TF-IDF:")
evaluate_model(nb_tfidf, X_test_tfidf, y_test_tfidf, feature_name="TF-IDF")

print("Bewertung für Bag of Words:")
evaluate_model(nb_bow, X_test_bow, y_test_bow, feature_name="BoW")

print("Bewertung für BoW und TF-IDF mit IG:")
evaluate_model(nb_ig_tfidf, X_test_ftfidf, y_test_ftfidf, feature_name="IG+TF-IDF")
evaluate_model(nb_ig_bow, X_test_fbow, y_test_fbow, feature_name="IG+BoW")

# Logistische Regression
print("LOGISTISCHE REGRESSION")
print("Bewertung für Punktuations-Features:")
evaluate_model(lr_punkt, X_test_punkt, y_test_punkt, feature_name="Punctuation_Features")

print("Bewertung für POS-Features:")
evaluate_model(lr_pos, X_test_pos, y_test_pos, feature_name="POS_Features")

print("Bewertung für Dependency-Features:")
evaluate_model(lr_dep, X_test_dep, y_test_dep, feature_name="Dependency_Features")

#print("Bewertung für LDA:")
evaluate_model(lr_lda, X_test_lda, y_test_lda, feature_name="LDA")

print("Bewertung für NER-Features:")
evaluate_model(lr_ner, X_test_ner, y_test_ner, feature_name="NER_Entities")

print("Bewertung für Emolex-Features:")
evaluate_model(lr_emot, X_test_emot, y_test_emot, feature_name="Emolex")

print("Bewertung für TF-IDF:")
evaluate_model(lr_tfidf, X_test_tfidf, y_test_tfidf, feature_name="TF-IDF")

print("Bewertung für Bag of Words:")
evaluate_model(lr_bow, X_test_bow, y_test_bow, feature_name="BoW")

print("Bewertung für BoW und TF-IDF mit IG:")
evaluate_model(lr_ig_tfidf, X_test_ftfidf, y_test_ftfidf, feature_name="IG+TF-IDF")
evaluate_model(lr_ig_bow, X_test_fbow, y_test_fbow, feature_name="IG+BoW")

# Support Vector Machines
print("SUPPORT VECTOR MACHINES")
print("Bewertung für Punktuations-Features:")
evaluate_model(svm_punkt, X_test_punkt, y_test_punkt, feature_name="Punctuation_Features")

print("Bewertung für POS-Features:")
evaluate_model(svm_pos, X_test_pos, y_test_pos, feature_name="POS_Features")

print("Bewertung für Dependency-Features:")
evaluate_model(svm_dep, X_test_dep, y_test_dep, feature_name="Dependency_Features")

print("Bewertung für LDA:")
evaluate_model(svm_lda, X_test_lda, y_test_lda, feature_name="LDA")

print("Bewertung für NER-Features:")
evaluate_model(svm_ner, X_test_ner, y_test_ner, feature_name="NER_Entities")

print("Bewertung für Emolex-Features:")
evaluate_model(svm_emot, X_test_emot, y_test_emot, feature_name="Emolex")

print("Bewertung für TF-IDF:")
evaluate_model(svm_tfidf, X_test_tfidf, y_test_tfidf, feature_name="TF-IDF")

print("Bewertung für Bag of Words:")
evaluate_model(svm_bow, X_test_bow, y_test_bow, feature_name="BoW")

print("Bewertung für BoW und TF-IDF mit IG:")
evaluate_model(svm_ig_tfidf, X_test_ftfidf, y_test_ftfidf, feature_name="IG+TF-IDF")
evaluate_model(svm_ig_bow, X_test_fbow, y_test_fbow, feature_name="IG+BoW")

# Kombinierte modelle, der Klassifikatorenset wurde immer wieder geändert, damit ein optimaler gefunden werden kann
print("KOMBINIERTE KLASSIFIKATOREN")
print("Bewertung für VotingClassifier:")
evaluate_model(voting_ensemble, X_test_combined, y_test_combined, feature_name="Combined Features") # einer der Besten mit NB
print("Bewertung für BaggingClassifier:")
evaluate_model(bagging_ensemble, X_test_combined, y_test_combined, feature_name="Bagging Combined Features") # einer der Besten mit NB
print("Bewertung für StackingClassifier:")
evaluate_model(stacking_ensemble, X_test_combined, y_test_combined, feature_name="Stacking Combined Features") # einer der Besten mit SVM
print("Bewertung für den Kombinierten Klassifikator:")
evaluate_model(nb_combined, X_test_combined, y_test_combined, feature_name="Combined") # einer der Besten

# Hyperparameter-Raster für GridSearch bei RF
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [None, 10, 20, 30, 40],
    'min_samples_split': [2, 3, 5],
    'min_samples_leaf': [1, 2, 3, 5],
    'class_weight': ['balanced', None],
    'criterion': ['gini', 'entropy']
}
# bei NB
param_grid_nb = {
    'alpha': [0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0], 
    'force_alpha': [True, False], 
    'fit_prior': [True, False],
    'norm': [True, False]
}
# bei LR
param_grid_lr = {
    'penalty': ['l1', 'l2', 'elasticnet', 'none'],  # Verschiedene Regularisierungen
    'tol': [1e-4, 1e-3, 1e-2],                      # Toleranzwerte für Konvergenz
    'C': [0.01, 0.1, 1.0, 10.0],                    # Inverser Regularisierungsparameter
    'solver': ['lbfgs', 'liblinear', 'saga'],       # Optimierungsalgorithmen
    'max_iter': [500, 700, 100, -1]                   # Maximale Anzahl der Iterationen
    #'multi_class': ['auto', 'ovr', 'multinomial'],  # Multiklassenhandling
}

# bei SVC
param_grid_svm = {
    'C': [0.1, 1, 10, 100],  # Regularisierungsparameter
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid'],  # Auswahl der Kernels
    'gamma': ['scale', 'auto', 0.01, 0.001],  # Kernel-Koeffizient
    'shrinking': [True, False],  # Schrinking-Heuristik
    'probability': [True, False],  # Wahrscheinlichkeitsschätzung aktivieren
    'tol': [1e-3, 1e-4],  # Toleranz für die Konvergenz
    'max_iter': [1000, 5000, -1],  # Maximale Iterationen (-1 = unbegrenzt)
    'decision_function_shape': ['ovo', 'ovr'],  # Entscheidungsfunktion
}

# Funktion zum Tuning für Random Forest
def tune_random_forest(X_train, y_train, X_test, y_test):
    rf = RandomForestClassifier(random_state=42)
    # GridSearchCV einrichten
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                               scoring='f1_weighted', cv=5, n_jobs=-1, verbose=2)
    # Tuning durchführen
    grid_search.fit(X_train, y_train)

    print("Beste Parameter:", grid_search.best_params_)
    print("Bester F1-Score:", grid_search.best_score_)
    
    # Modell mit besten Parametern trainieren und auf Testdaten auswerten
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("Test Genauigkeit:", accuracy)
    print("Test F1-Score:", f1)

# Funktion zum Tuning für Naive Bayes
def tune_nb(X_train, y_train, X_test, y_test):
    nb = ComplementNB()
    grid_search = GridSearchCV(estimator=nb, param_grid=param_grid_nb, 
                               scoring='f1_weighted', cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    print("Beste Parameter:", grid_search.best_params_)
    print("Bester F1-Score:", grid_search.best_score_)
    
    best_nb = grid_search.best_estimator_
    y_pred = best_nb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("Test Genauigkeit:", accuracy)
    print("Test F1-Score:", f1)

# Funktion zum Tuning für Logistische Regression
def tune_lr(X_train, y_train, X_test, y_test):
    log_reg = LogisticRegression(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(estimator=log_reg, param_grid=param_grid_lr, 
                               scoring='f1_weighted', cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    print("Beste Parameter:", grid_search.best_params_)
    print("Bester F1-Score:", grid_search.best_score_)
    
    best_log_reg = grid_search.best_estimator_
    y_pred = best_log_reg.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("Test Genauigkeit:", accuracy)
    print("Test F1-Score:", f1)

# Funktion zum Tuning für SVM
def tune_svm(X_train, y_train, X_test, y_test):
    svm_model = SVC(random_state=42, class_weight='balanced')
    grid_search = GridSearchCV(estimator=svm_model, param_grid=param_grid_svm, 
                               scoring='f1_weighted', cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)
    
    print("Beste Parameter:", grid_search.best_params_)
    print("Bester F1-Score (Train):", grid_search.best_score_)

    best_svm = grid_search.best_estimator_
    y_pred = best_svm.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print("Test Genauigkeit:", accuracy)
    print("Test F1-Score:", f1)

# Hyperparameter-Tuning für jeden Datensatz durchführen
#print("Tuning für Punktuations-Features:")
#tune_random_forest(X_train_punkt, y_train_punkt, X_test_punkt, y_test_punkt)
#tune_nb(X_train_punkt, y_train_punkt, X_test_punkt, y_test_punkt)
#tune_lr(X_train_punkt, y_train_punkt, X_test_punkt, y_test_punkt)
#tune_svm(X_train_punkt, y_train_punkt, X_test_punkt, y_test_punkt)

#print("\nTuning für POS-Features:")
#tune_random_forest(X_train_pos, y_train_pos, X_test_pos, y_test_pos)
#tune_nb(X_train_pos, y_train_pos, X_test_pos, y_test_pos)
#tune_lr(X_train_pos, y_train_pos, X_test_pos, y_test_pos)
#tune_svm(X_train_pos, y_train_pos, X_test_pos, y_test_pos)

#print("\nTuning für Dependency-Features:")
#tune_random_forest(X_train_dep, y_train_dep, X_test_dep, y_test_dep)
#tune_nb(X_train_dep, y_train_dep, X_test_dep, y_test_dep)
#tune_lr(X_train_dep, y_train_dep, X_test_dep, y_test_dep)
#tune_svm(X_train_dep, y_train_dep, X_test_dep, y_test_dep)

#print("\nTuning für LDA-Features:")
#tune_random_forest(X_train_lda, y_train_lda, X_test_lda, y_test_lda)
#tune_nb(X_train_lda, y_train_lda, X_test_lda, y_test_lda)
#tune_lr(X_train_lda, y_train_lda, X_test_lda, y_test_lda)
#tune_svm(X_train_lda, y_train_lda, X_test_lda, y_test_lda)

#print("\nTuning für NER-Features:")
#tune_random_forest(X_train_ner, y_train_ner, X_test_ner, y_test_ner)
#tune_nb(X_train_ner, y_train_ner, X_test_ner, y_test_ner)
#tune_lr(X_train_ner, y_train_ner, X_test_ner, y_test_ner)
#tune_svm(X_train_ner, y_train_ner, X_test_ner, y_test_ner)

#print("\nTuning für Emolex-Features:")
#tune_random_forest(X_train_emot, y_train_emot, X_test_emot, y_test_emot)
#tune_nb(X_train_emot, y_train_emot, X_test_emot, y_test_emot)
#tune_lr(X_train_emot, y_train_emot, X_test_emot, y_test_emot)
#tune_svm(X_train_emot, y_train_emot, X_test_emot, y_test_emot)

#print("\nTuning für TF-IDF:")
#tune_random_forest(X_train_tfidf, y_train_tfidf, X_test_tfidf, y_test_tfidf)
#tune_nb(X_train_tfidf, y_train_tfidf, X_test_tfidf, y_test_tfidf)
#tune_lr(X_train_tfidf, y_train_tfidf, X_test_tfidf, y_test_tfidf)
#tune_svm(X_train_tfidf, y_train_tfidf, X_test_tfidf, y_test_tfidf)

#print("\nTuning für BoW:")
#tune_random_forest(X_train_bow, y_train_bow, X_test_bow, y_test_bow)
#tune_nb(X_train_bow, y_train_bow, X_test_bow, y_test_bow)
#tune_lr(X_train_bow, y_train_bow, X_test_bow, y_test_bow)
#tune_svm(X_train_bow, y_train_bow, X_test_bow, y_test_bow)

#print("\nTuning für IG TF-IDF:")
#tune_random_forest(X_train_ftfidf, y_train_ftfidf, X_test_ftfidf, y_test_ftfidf)
#tune_nb(X_train_ftfidf, y_train_ftfidf, X_test_ftfidf, y_test_ftfidf)
#tune_lr(X_train_ftfidf, y_train_ftfidf, X_test_ftfidf, y_test_ftfidf)
#tune_svm(X_train_ftfidf, y_train_ftfidf, X_test_ftfidf, y_test_ftfidf)

#print("\nTuning für IG BoW:")
#tune_random_forest(X_train_fbow, y_train_fbow, X_test_fbow, y_test_fbow)
#tune_nb(X_train_fbow, y_train_fbow, X_test_fbow, y_test_fbow)
#tune_lr(X_train_fbow, y_train_fbow, X_test_fbow, y_test_fbow)
#tune_svm(X_train_fbow, y_train_fbow, X_test_fbow, y_test_fbow)

#print("\nTuning für das kombinierte Modell:")
#tune_random_forest(X_train_combined, y_train_combined, X_test_combined, y_test_combined)
#tune_nb(X_train_combined, y_train_combined, X_test_combined, y_test_combined)
#tune_lr(X_train_combined, y_train_combined, X_test_combined, y_test_combined)
#tune_svm(X_train_combined, y_train_combined, X_test_combined, y_test_combined)

''' Wichtigkeit der einzelnen Features'''
'''# Extraktion der Feature-Namen bei BoW und TF-IDF
tfidf_feature_names = df_tfidf.drop(columns=['Label', 'Titel']).columns
bow_feature_names = df_bow.drop(columns=['Label', 'Titel']).columns

importances = rf_combined.feature_importances_ # wurde für rf_combined getestet
# Top Features
print("Feature Importances:", importances)
indices = np.argsort(importances)[::-1]
feature_names = (
    list(tag_to_index.keys()) +                     # POS-Tags
    [f'Punctuation_{i}' for i in range(len(X_punkt[0]))] +  # Punctuation
    list(dep_to_index.keys()) +                     # Dependency-Features
    list(emotion_to_index.keys()) + # LDA-Features
    [f'NER_{ent}_{label}' for (ent, label) in ner_to_index.keys()] +  # NER-Entities
    list(emotion_to_index.keys()) +                 # Emotion-Features
    list(df_tfidf.drop(columns=['Label', 'Titel']).columns) +  # TF-IDF
    list(df_bow.drop(columns=['Label', 'Titel']).columns)      # Bag-of-Words
)

# Test, ob die Länge der Feature-Namen korrekt ist
print(f"Anzahl der Feature-Namen: {len(feature_names)}")
print(f"Anzahl der Features in X_combined: {X_combined.shape[1]}")

# Top-10-Features anzeigen
print("Top 10 wichtigste Features mit Namen:")
for i in indices[:10]:  # Top 10 Features
    print(f"{feature_names[i]} - Importance: {importances[i]:.4f}")'''

'''Visualisierung'''

def split_labels(labels):
    # Labels auf mehrere Zeilen aufteilen basierend auf dem '&'-Zeichen für bessere Lesbarkeit
    return [label.replace('&', '&\n') for label in labels]

def plot_confusion_matrix(model, X_test, y_test, filename=None):
    y_pred = model.predict(X_test)
    le = LabelEncoder()
    y_test_encoded = le.fit_transform(y_test)
    y_pred_encoded = le.transform(y_pred)
    cm = confusion_matrix(y_test_encoded, y_pred_encoded)
    class_labels = le.classes_  # Holen der originalen Klassenlabels
    split_class_labels = split_labels(class_labels)  # Teilen der Labels
    # Plotten
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='viridis', xticklabels=split_class_labels, yticklabels=split_class_labels, square=True)
    plt.title(f'Confusion Matrix des Modells {model.__class__.__name__} für Allgemeines Deutsches Kommersbuch\n', loc='center', fontweight='bold')
    plt.text(0, 0, '', fontsize=16)  # Leerzeile
    plt.xlabel('Vorhergesagte Klassenzugehörigkeiten', fontweight='bold')
    plt.ylabel('Tatsächliche Klassenzugehörigkeiten', fontweight='bold')
    plt.xticks(rotation=45, ha='right')  # ha='right' für horizontale Ausrichtung
    plt.yticks(rotation=0)  # y-Ticks nicht drehen
    plt.tight_layout() # automatische Anpassung
    if filename is not None:
        plt.savefig(filename, format='png', bbox_inches='tight')
    plt.show()

def plot_tsne_classification(model, X_test, y_test, filename=None):
    y_pred = model.predict(X_test)
    le = LabelEncoder()
    y_test_encoded = le.fit_transform(y_test)
    y_pred_encoded = le.transform(y_pred)
    tsne = TSNE(n_components=2, random_state=42)
    X_test_tsne = tsne.fit_transform(X_test)
    class_labels = le.classes_ 
    viridis = plt.get_cmap('viridis', len(class_labels))
    # Plotten
    plt.figure(figsize=(12, 8))
    scatter1 = plt.scatter(X_test_tsne[:, 0], X_test_tsne[:, 1], c=y_test_encoded, 
                           cmap='viridis', s=200, marker='o', linewidth=0, label='Tatsächliche Klassen', alpha=0.7, edgecolor='k')
    scatter2 = plt.scatter(X_test_tsne[:, 0], X_test_tsne[:, 1], c=y_pred_encoded, 
                           cmap='viridis', s=100, marker='.', linewidth=0, label='Vorhergesagte Klassen')
    plt.title(f'                                t-SNE des Modells {model.__class__.__name__} für Allgemeines Deutsches Kommersbuch\n', loc='center',fontweight='bold')
    plt.text(0, 0, '', fontsize=16)
    cbar1 = plt.colorbar(scatter1, ax=plt.gca(), aspect=30, fraction=0.046,)
    cbar1.set_label('Tatsächliche Klassen', fontweight='bold') 
    cbar2 = plt.colorbar(scatter2, ax=plt.gca(), aspect=30, fraction=0.046, pad=0.04)
    cbar2.set_label('Vorhergesagte Klassen', fontweight='bold')
    for i, label in enumerate(class_labels):
        y_pos = i / (len(class_labels) - 1) # Normalisiert die Position auf den Bereich [0, 1]
        if i == 0:
            label_y_position = y_pos + 0.08  # Erhöht die Position des ersten Labels
        else:
            label_y_position = y_pos + 0.8 * i
        cbar1.ax.text(3.8, label_y_position, label, va='center', ha='left', fontsize=10, fontweight='bold', color=viridis(i))
    plt.xticks([])
    plt.yticks([])  
    plt.legend(loc='upper left')
    plt.gca().set_aspect('equal', adjustable='box')
    if filename is not None:
        plt.savefig(filename, format='png', bbox_inches='tight')
    plt.show()

plot_confusion_matrix(voting_ensemble, X_test_combined, y_test_combined, filename='voting_cm_plot.png')
plot_confusion_matrix(stacking_ensemble, X_test_combined, y_test_combined, filename='stacking_cm_plot.png')
plot_confusion_matrix(nb_bow, X_test_bow, y_test_bow, filename='nbbow_cm_plot.png')
plot_tsne_classification(voting_ensemble, X_test_combined, y_test_combined, filename='voting_tsne_plot.png')
plot_tsne_classification(stacking_ensemble, X_test_combined, y_test_combined, filename='stacking_tsne_plot.png')
plot_tsne_classification(nb_bow, X_test_bow, y_test_bow, filename='nbbow_tsne_plot.png')