# -*- coding: utf-8 -*-
"""Dieser Code zum Extrahieren und Abspeichern der Features für die Klassifizierung wurde auf Google Colab auf T4 GPU ausgeführt"""

import re
import random
import json
import requests
import nltk
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
import pandas as pd
import numpy as np
import stanza
import gensim
import cltk
import spacy
spacy.prefer_gpu()
import logging
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
from nltk import pos_tag, word_tokenize, WordNetLemmatizer
from gensim import corpora
from cltk.lemmatize.lat import LatinBackoffLemmatizer
from cltk.data.fetch import FetchCorpus
corpus_downloader = FetchCorpus(language='lat')
corpus_downloader.import_corpus('lat_models_cltk')
nlp_deu = spacy.load('de_dep_news_trf')
nlp_lat = LatinBackoffLemmatizer()
nlp_de = stanza.Pipeline('de', processors='tokenize,mwt,pos,lemma,depparse,ner', use_gpu=True)  # Deutsch
nlp_la = stanza.Pipeline('la', processors='tokenize,mwt,pos,lemma,depparse', use_gpu=True)  # Latein
from google.colab import files
uploaded = files.upload()

# ISO-Stopwordliste von Diaz, Gene: https://github.com/stopwords-iso/stopwords-iso
def load_stopwords(languages=["de", "fr", "la"]):
    url = "https://raw.githubusercontent.com/stopwords-iso/stopwords-iso/master/stopwords-iso.json"
    response = requests.get(url)

    if response.status_code == 200:
        stopwords_json = response.json()
        stopwords = []
        for lang in languages:
            stopwords += stopwords_json.get(lang, [])
        return stopwords
    else:
        print(f"Fehler beim Laden der Stopwörter. Statuscode: {response.status_code}")
        return []

# Entfernen der Stopwörter, Ziffern und Zeichen
def clean_text(text, stopwords):
    # Entfernen von Ziffern und Zeichen (außer Buchstaben und Leerzeichen)
    text = re.sub(r'[^\w\s]', '', text)  # Entfernt Sonderzeichen
    text = re.sub(r'\d+', '', text)  # Entfernt Ziffern
    text = re.sub(r'\bec\.', '', text) # Entfernen von "ec."
    text = re.sub(r'\bec', '', text) # Entfernen von "ec"
    text = re.sub(r'\betc\.', '', text) # Entfernen von "etc."
    text = re.sub(r'\|:|:\|', '', text)  # Entfernt Wiederholungszeichen aller Art
    text = re.sub(r':¦', '', text)
    text = re.sub(r'¦:', '', text)

    words = text.split()
    cleaned_text = " ".join([word for word in words if word.lower() not in stopwords]) # Stopwörter werden aus dem Text entfernt
    return cleaned_text

# Lemmatisierung des Textes
def lemmatize_text(text):
    text = str(text)
    doc = nlp_lat(text) if " est " in text or " et " in text else nlp_deu(text)
    lemmatized_sentence = ' '.join([x.lemma_ for x in doc])
    return lemmatized_sentence

# Datei einlesen
def read_txt_file(filename, stopwords):
    labeled_songs = []
    with open(filename, "r", encoding="utf-8") as file:
        label = None
        title = None
        text = []

        for line in file:
            line = line.strip()
            if line.startswith("Label:"):
                if title and text:
                    original_text = " ".join(text).strip()
                    cleaned_text = clean_text(original_text, stopwords)
                    lemmatized_text = lemmatize_text(cleaned_text)
                    labeled_songs.append((label, title, original_text, cleaned_text, lemmatized_text))
                label = line.split("Label:")[1].strip()
                title = None
                text = []
            elif line.startswith("Titel:"):
                title = line.split("Titel:")[1].strip()
            elif line.startswith("Text:"):
                continue
            else:
                text.append(line)

        # Letztes Lied hinzufügen
        if title and text:
            original_text = " ".join(text).strip()
            cleaned_text = clean_text(original_text, stopwords)
            lemmatized_text = lemmatize_text(cleaned_text)
            labeled_songs.append((label, title, original_text, cleaned_text, lemmatized_text))

    return labeled_songs # Format: Liste [(Label, Titel, Originaltext, Bereinigt, Lemmatisiert), (Label, Titel, Originaltext, Bereinigt, Lemmatisiert)...]

# Liedertexte und Labels vorbereiten
def prepare_data(labeled_songs):
    labels = []
    titles = []
    original_texts = []
    cleaned_texts = []
    lemmatized_texts = []

    for label, title, original_text, cleaned_text, lemmatized_text in labeled_songs:
        labels.append(label)
        titles.append(title)
        original_texts.append(original_text)
        cleaned_texts.append(cleaned_text)
        lemmatized_texts.append(lemmatized_text)

    return labels, titles, original_texts, cleaned_texts, lemmatized_texts # Format: 5 Listen: [Label, Label, Label...] [Titel, Titel, Titel...] [Text, Text, Text...] ...

# Erstellen eines DataFrames
def create_dataframe(labels, titles, original_texts, cleaned_texts, lemmatized_texts):
    df = pd.DataFrame({
        'Label': labels,
        'Titel': titles,
        'Originaltext': original_texts,
        'Bereinigt': cleaned_texts,
        'Lemmatisiert': lemmatized_texts
    })
    return df # Format: Dataframe mit folgenden Spralten: Label, Titel, Originaltext, Bereinigt, Lemmatisiert

''' Features auf dem Basis des Originaltextes '''
# Funktion zum Zählen der Interpunktionszeichen und Erstellen der Merkmale
def extract_and_count_punctuation(df):
    punkt_counts = {}

    for index, row in df.iterrows():
        title = row['Titel']
        text = row['Originaltext']
        punctuation_counts = {
            'periods': text.count('.'),
            'commas': text.count(','),
            'exclamations': text.count('!'),
            'questions': text.count('?'),
            'semicolons': text.count(';'),
            'colons': text.count(':'),
            'quotation_marks': text.count('"') + text.count('„') + text.count('“'),  # Anführungszeichen jeder Art
            'dashes': text.count('—') + text.count('-'),  # Striche
            'parentheses': text.count('(') + text.count(')'),  # Klammern
            'ellipses': text.count('...') + text.count('..'),  # Auslassungspunkte jeder Art
            'two_exclamations': text.count('!!'),
            'three_exclamations': text.count('!!!'),  # Doppelte und dreifache Ausrufezeichen
            'repeat': text.count('ec.') + text.count('etc.') + text.count(':|') + text.count('|:') + text.count('¦:') + text.count(':¦')  # Wiederholungsanweisungen
        }

        # Erstellen einer Liste von Tupeln (Feature, Count)
        punkt_list = [(key, value) for key, value in punctuation_counts.items()]

        # Speichern im Dictionary
        punkt_counts[title] = punkt_list

    return punkt_counts # Format: Dictionary {'1. Titel': [('Feature1', Zahl), ('Feature2', Zahl), ...], '2. Titel': [('Feature1', Zahl), ('Feature2', Zahl)...], ... }

# Funktion für POS-Tagging
def pos_tagging(df, nlp_de, nlp_la):
    pos_counts = {}

    for index, row in df.iterrows():
        title = row['Titel']
        text = row['Originaltext']
        nlp = nlp_la if " est " in text or " et " in text else nlp_de
        doc = nlp(text)
        tag_counts = Counter()

        # Sätze und Tokens im Dokument durchlaufen
        for sentence in doc.sentences:
            for word in sentence.words:
                tag_counts[word.upos] += 1  # upos = universelle POS-Tags
        pos_list = [(pos, count) for pos, count in tag_counts.items()]
        pos_counts[title] = pos_list

    return pos_counts # Format: Dictionary {'1. Titel': [('POS1', Zahl), ('POS2', Zahl), ...], '2. Titel': [('POS1', Zahl), ('POS2', Zahl...], ... }

# Funktion zur Abhängigkeitsanalyse
def dependency_parsing(df):
    dependencies = {}

    for index, row in df.iterrows():
        title = row['Titel']
        text = row['Originaltext']

        # Prüfen, ob der Text lateinisch ist
        doc = nlp_la(text) if " est " in text or " et " in text else nlp_de(text)

        # Abhängigkeitsrelationen direkt aus allen Wörtern des Dokuments zählen
        relation_counter = Counter([word.deprel for sentence in doc.sentences for word in sentence.words])

        # Speichern als Liste von Tupeln
        dependencies[title] = list(relation_counter.items())

    return dependencies # Format: Dictionary {'1. Titel': [('Abhängigkeitsart1', Zahl), ('Abhängigkeitsart2', Zahl), ...], '2. Titel': [('Abhängigkeitsart1', Zahl), ('Abhängigkeitsart2', Zahl)...], ... }

''' Features auf dem Basis des bereinigten Textes: ohne Stoppwörter, ohne Zeichen, ohne Ziffern, ohne musikalische Anweisungen '''
# Named Entity Recognition
def ner_entity_count(df):
    entity_counts = {}

    for index, row in df.iterrows():
        title = row['Titel']
        text = row['Bereinigt']
        doc = nlp_de(text)
        entities = Counter((ent.text, ent.type) for ent in doc.entities)
        entity_counts[title] = list(entities.items())

    return entity_counts # Format: Dictionary {'1. Titel': [(('Feature1', NE), Zahl), (('Feature2', NE), Zahl) ...], '2. Titel': [(('Feature1', NE), Zahl), (('Feature2', NE), Zahl)...], ... }

''' Features auf dem Basis des lemmatisierten Textes '''
# Latent Dirichlet Allocation
def lda_feature_extraction(df):

    # Tokenisierung der lemmatisierten Texte
    texts = [
        row.Lemmatisiert.split() if isinstance(row.Lemmatisiert, str) else row.Lemmatisiert
        for row in df.itertuples(index=False)
    ]

    # Wörterbuch und Korpus erstellen
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]

    # LDA-Modell trainieren
    lda_model = gensim.models.LdaModel(corpus=corpus, id2word=dictionary, num_topics=5, random_state=42)

    # Thema-Titel für alle Topics
    topic_titles = {
        f'Topic {i + 1}': " | ".join([word for word, _ in lda_model.show_topic(i, topn=3)])
        for i in range(lda_model.num_topics)
    }

    # Konsistenzprüfung zwischen DataFrame und Korpus
    if len(df) != len(corpus):
        raise ValueError("Anzahl der Zeilen im DataFrame und Korpus stimmen nicht überein.")

    # LDA-Features pro Dokument berechnen und zuordnen
    lda_features = {
        row.Titel: [
            (topic_titles[f'Topic {topic + 1}'], round(prob, 4))
            for topic, prob in lda_model.get_document_topics(corpus[i], minimum_probability=0.0)
        ]
        for i, row in enumerate(df.itertuples(index=False))
    }

    return lda_features # Format: Dictionary {'1. Titel': [('Thema1', Gewichtung), ('Thema2', Gewichtung), ...], '2. Titel': [('Thema1', Gewichtung), ('Thema2', Gewichtung)...], ... }

# Emolex mit der heruntergeladenem Emotionslexikon für deutsch aus https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
def load_emolex():
    uploaded = files.upload()
    emolex_path = "German-NRC-EmoLex.txt"
    emolex_df = pd.read_csv(emolex_path, delimiter='\t')

    # Wörterbuch für deutsche Wörter und deren Emotionen
    emolex_dict = {}
    for index, row in emolex_df.iterrows():
        german_word = row['German Word']
        emotions = row.drop(['English Word', 'German Word']).to_dict()
        # Nur Wörter mit mindestens einer Emotion hinzufügen
        if any(emotions.values()):
            emolex_dict[german_word] = {k: int(v) for k, v in emotions.items()}

    return emolex_dict

# Funktion zur Berechnung der Emotionen pro Text
def calculate_emotions(df, emolex_dict, text_column='Lemmatisiert'):
    emotions_per_text = {}

    for index, row in df.iterrows():
        title = row['Titel']
        text = row[text_column].split()

        # Zählt die Emotionen pro Text
        emotion_counter = defaultdict(int)

        for word in text:
            if word in emolex_dict:
                emotions = emolex_dict[word]
                for emotion, value in emotions.items():
                    emotion_counter[emotion] += value

        # Normiert die Emotionen nach der Anzahl der Wörter
        total_words = len(text)
        normalized_emotions = {emotion: round(count / total_words, 4) for emotion, count in emotion_counter.items()}

        # Speichert die Emotionen im gewünschten Format
        emotions_per_text[title] = sorted(normalized_emotions.items(), key=lambda x: x[1], reverse=True)

    return emotions_per_text # Format: Dictionary {'1. Titel': [('Emotion1', Gewichtung), ('Emotion2', Gewichtung), ...], '2. Titel': [('Emotion1', Gewichtung), ('Emotion2', Gewichtung)...], ... }

# Funktion zur Berechnung des Informationsgewinns für Wörter
def calculate_ig(df, text_column='Lemmatisiert', label_column='Label', top_n=50):
    # CountVectorizer für alle Texte
    vectorizer = CountVectorizer()
    X = vectorizer.fit_transform(df[text_column])
    y = df[label_column]
    # Berechnung des IG für das gesamte Vokabular
    ig = mutual_info_classif(X, y, discrete_features=True)
    # DataFrame mit allen Wörtern und ihren IG-Werten
    ig_df = pd.DataFrame({'Wort': vectorizer.get_feature_names_out(), 'Information Gain': ig})
    ig_df['Information Gain'] = ig_df['Information Gain'].round(5) # auf 5 Nachkommastellen gerundet
    # Sortierung nach IG absteigend
    ig_df = ig_df.sort_values(by='Information Gain', ascending=False)
    # Liste zur Speicherung der Top-N-Wörter pro Kategorie
    result = []
    # Iterieration über jede Kategorie, Top-N-Wörter nach IG werden für jede Kategorie bestimmt
    for category in df[label_column].unique():
        # Filtern der Texte nach Kategorie
        df_category = df[df[label_column] == category]
        # Wörter aus dieser Kategorie, die im globalen IG-DataFrame vorkommen
        words_in_category = set(' '.join(df_category[text_column]).split())
        category_ig_df = ig_df[ig_df['Wort'].isin(words_in_category)]
        # Extraktion der Top-N-Wörter für die Kategorie
        top_words = category_ig_df.head(top_n)
        result.append(top_words)
        top_words['Label'] = category
    # Alle Ergebnisse in einem DataFrame
    final_ig_df = pd.concat(result).reset_index(drop=True)
    final_ig_df = final_ig_df[['Label', 'Wort', 'Information Gain']]

    return final_ig_df

# Format: Dataframe, mit Separator ';',  insgesamt 70 Spalten mit den Wörtern, da 10 Wörter je Kategorie
#	Label Wort        IG
# X     Wort1	1	    0
# X     Wort1	2	    2
# Y     Wort1	3	    0

# Funktion für die Erstellung des BoW pro Kategorie mit jeweils 10 Wörtern
def create_bow_per_category(df, text_column='Lemmatisiert', label_column='Label', title_column='Titel', top_n=50):
    # Liste für BoW
    bow_rows = []
    # Iterieren durch jede Kategorie
    for label in df[label_column].unique():
        # Filtern der Texte nach Kategorie
        label_df = df[df[label_column] == label]
        # CountVectorizer für die aktuelle Kategorie
        vectorizer = CountVectorizer(max_features=top_n)  # Begrenzung auf Top-N häufigsten Wörter
        # Anpassung von dem Vektorizer auf die Texte der Kategorie und BoW-Features
        X_bow = vectorizer.fit_transform(label_df[text_column])
        feature_names = vectorizer.get_feature_names_out()
        # Iterieration über jede Kategorie, BoW-Daten werden extrahiert
        for i, (index, row) in enumerate(label_df.iterrows()):
            title = row[title_column]
            bow_vector = X_bow[i].toarray().flatten()
            # Dictionary für die aktuelle Zeile
            bow_row = {'Label': label, 'Titel': title}
            bow_row.update({feature_names[j]: bow_vector[j] for j in range(len(feature_names))})
            bow_rows.append(bow_row)
    # DataFrame aus den gesammelten Zeilen
    combined_bow_df = pd.DataFrame(bow_rows).fillna(0)

    return combined_bow_df

# Funktion für die Erstellung des TF-IDFs pro Kategorie mit jeweils 10 Wörtern
def create_tfidf_per_category(df, text_column='Lemmatisiert', label_column='Label', title_column='Titel', n=50):
    # Liste für TF-IDF
    tfidf_rows = []
    # Iterieren durch jede Kategorie
    for label in df[label_column].unique():
        # Filtern der Texte nach Kategorie
        label_df = df[df[label_column] == label]
        # TfidfVectorizer für die Kategorie
        vectorizer = TfidfVectorizer(max_features=n)
        # Vektorizer wird angepasst und die Texte der Kategorie in TF-IDF-Features transformiert
        X_tfidf = vectorizer.fit_transform(label_df[text_column])
        feature_names = vectorizer.get_feature_names_out()
        # Iterieren durch jede Zeile im DataFrame für die aktuelle Kategorie
        for i, (index, row) in enumerate(label_df.iterrows()):
            title = row[title_column]
            tfidf_vector = X_tfidf[i].toarray().flatten()
            # Dictionary für die aktuelle Zeile
            tfidf_row = {'Label': label, 'Titel': title}
            tfidf_row.update({feature_names[j]: round(tfidf_vector[j], 5) for j in range(len(feature_names))})
            tfidf_rows.append(tfidf_row)
    # Finales DataFrame
    combined_tfidf_df = pd.DataFrame(tfidf_rows).fillna(0)

    return combined_tfidf_df

# Format bei BoW und TF-IDF: Dataframe, mit Separator ';', Spalten mit den einzigartigen Wörtern (d. h. ohne Duplikate von Wörtern, die in mehreren Kategorien vorkommen), da 10 Wörter je Kategorie
#	Label Titel Wort1	Wort2	Wort3	Wort4	Wort5
# X     1. Titel	1	    0	    0	    0	    1
# X     2. Titel	1	    2	    0	    0	    1
# Y     3. Titel	0	    0	    2	    1	    1 ...


# Hauptlogik
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    logging.info("Starte Programm ...")

    filename = "kommersbuch_output.txt"

    # Stopwörter
    stopwords = load_stopwords()

    # Datei einlesen und Stopwörter entfernen
    labeled_songs = read_txt_file(filename, stopwords)

    # Datenaufbereitung

    labels, titles, original_texts, cleaned_texts, lemmatized_texts = prepare_data(labeled_songs)

    # Dataframe
    logging.info("Erstelle DataFrame ...")
    df_kommersbuch = create_dataframe(labels, titles, original_texts, cleaned_texts, lemmatized_texts)

    # Features mit Interpunktion und musikalischen Anweisungen
    logging.info("Extrahiere Interpunktion und musikalischen Anweisungen ...")
    punkt_features = extract_and_count_punctuation(df_kommersbuch)

    # POS-Tagging
    logging.info("Bestimme Wortarten ...")
    tagged_texts = pos_tagging(df_kommersbuch, nlp_de=nlp_de, nlp_la=nlp_la)

    # Abhängigkeitsverhältnisse
    logging.info("Bestimme Abhängigkeiten ...")
    depend_texts = dependency_parsing(df_kommersbuch)

    # Named Entity Recognition
    logging.info("Bestimme Entitäten ...")
    ner_texts = ner_entity_count(df_kommersbuch)

    # Latent Dirichlet Allocation
    logging.info("Bestimme Wortkombinationen ...")
    lda_texts = lda_feature_extraction(df_kommersbuch)

    #Emolex
    logging.info("Bestimme Emitionen ...")
    emolex_dict = load_emolex()
    emotions_per_text = calculate_emotions(df_kommersbuch, emolex_dict)

    # Zu jedem Titel im DF werden die Features nach der Erstellung eingefügt und einschließend als CSV-Datei gespeichert
    df_kommersbuch['Label'] = df_kommersbuch['Label'].astype(str)
    df_kommersbuch['Punctuation_Features'] = df_kommersbuch['Titel'].map(punkt_features)
    df_kommersbuch['POS_Tags'] = df_kommersbuch['Titel'].map(tagged_texts)
    df_kommersbuch['Dependency_Features'] = df_kommersbuch['Titel'].map(depend_texts)
    df_kommersbuch['LDA_Topics'] = df_kommersbuch['Titel'].map(lda_texts)
    df_kommersbuch['NER_Entities'] = df_kommersbuch['Titel'].map(ner_texts)
    df_kommersbuch['Emotion_Features'] = df_kommersbuch['Titel'].map(emotions_per_text)
    df_kommersbuch.to_csv('df_kommersbuch.csv', sep=';', index=False)

    # IG, anschließend als CSV-Datei gespeichert
    ig_df = calculate_ig(df_kommersbuch)
    ig_df.to_csv('ig_kommersbuch.csv', sep=';', index=False)

    # BoW, anschließend als CSV-Datei gespeichert
    bow_df = create_bow_per_category(df_kommersbuch, text_column='Lemmatisiert', label_column='Label', title_column='Titel', top_n=50)
    bow_df.to_csv('bow_kommersbuch.csv', sep=';', index=False)

    # TF-IDF, anschließend als CSV-Datei gespeichert
    tfidf_df = create_tfidf_per_category(df_kommersbuch, text_column='Lemmatisiert', label_column='Label', title_column='Titel', n=50)
    tfidf_df.to_csv('tfidf_kommersbuch.csv', sep=';', index=False)

    logging.info("Programmdurchlauf erfolgreich")

    files.download('df_kommersbuch.csv')
    files.download('ig_kommersbuch.csv')
    files.download('bow_kommersbuch.csv')
    files.download('tfidf_kommersbuch.csv')