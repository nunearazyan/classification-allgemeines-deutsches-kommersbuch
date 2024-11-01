import requests
from bs4 import BeautifulSoup
import re

def extract_song_titles_and_texts(start_page, end_page):
    titles_and_texts = []
    last_title = None  # Speichert den letzten Titel für Fortsetzungen
    last_text = []  # Speichert den Text des letzten Titels

    for page_number in range(start_page, end_page + 1):
        url = f'https://de.wikisource.org/wiki/Allgemeines_Deutsches_Kommersbuch:{page_number}'
        response = requests.get(url)

        soup = BeautifulSoup(response.content, 'html.parser')

        # Suchen nach Liedtiteln (alle <span> mit ID)
        titles = soup.find_all('span', id=True)
        
        # Behandelt jeden Titel separat
        for title in titles:
            title_text = title.get_text(strip=True)
            title_text = re.sub(r'\s*\(.*?\)\s*', ' ', title_text)  # Entfernt Inhalt in Klammern
            title_text = re.sub(r'\*\).*$', '', title_text) # Entfernt Anmerkungen zu den Titeln
            title_text = re.sub(r'\s+', ' ', title_text)  # Entfernt überflüssige Leerzeichen
            
            song_text = []

            # Suche nach der nächsten Tabelle oder Gedicht
            for sibling in title.find_all_next():
                if sibling.name == 'div' and 'poem' in sibling.get('class', []):
                    # Extrahiere den Gedichttext
                    text = sibling.get_text(strip=False)
                    # Bereinige den Text
                    text = re.sub(r'\[.*?\]', '', text)  # Entfernt eckige Klammern und deren Inhalt
                    text = re.sub(r'\s*(\d+\.?)', '', text)  # Entfernt Ziffern
                    text = re.sub(r'(?<=\w)-(?=\w)', '', text) # Zusammenfügen von Wörtern, die durch "-" getrennt sind
                    text = re.sub(r'\n\s*\n|\n+', ' ', text)  # Entfernt Zeilenumbrüche
                    text = re.sub(r'\s+', ' ', text)  # Entfernt mehrere Leerzeichen
                    text = re.sub(r'=\s*', '', text)  # Zusammenfügen von Wörtern, die durch ein Gleichheitszeichen getrennt sind
                    song_text.append(text.strip()) # bereinigte Texte werden zusammegefügt, wenn zwischen beiden Textabschnitten kein Titelabschnitt ist

                elif sibling.name == 'span' and sibling.get('id'):  # Wird nächster Titel gefunden, wird die Extraktion der Liedtexte von neu begonnen, ansonsten werden die Texte zu demselben Titel weoterhin bezogen. Dies gelingt, weil einige Texte sich über ehrere HTML-Seiten erstrecken (d. h. in mehreren <div> enthalten sind), wobei sie im Ergebnis unter demselben Titel erscheinen sollen
                    break  # Beendet die Suche, wenn der nächste Titel gefunden wird

            # Texte werden mit dem Titel gespeichert
            if song_text:
                full_text = " ".join(song_text).strip()
                titles_and_texts.append((title_text, full_text))

    return titles_and_texts

def label_songs(songs):
    labeled_songs = []
    for title, text in songs:
        # Die Nummer aus dem Titel, die darin auf Wikisource schon enthalten sind, werden extrahiert
        match = re.match(r'^\d+\.\s*(.*)', title)
        if match:
            title_number = int(match.group(0).split('.')[0])
            # Das Label wird basierend auf der Nummer bestimmt, wobei zur Referenz Folgende Seite dient, wo die Kategorien ebenfalls benannt werden: https://de.wikisource.org/wiki/Allgemeines_Deutsches_Kommersbuch 
            if 1 <= title_number <= 135:
                label = "Vaterlands & Heimatlieder"
            elif 136 <= title_number <= 230:
                label = "Festgesänge & Gemeinschaftslieder"
            elif 231 <= title_number <= 322:
                label = "Jugend & Erinnerung"
            elif 323 <= title_number <= 431:
                label = "Liebe, Wein & Wandern"
            elif 432 <= title_number <= 557:
                label = "Volkslieder"
            elif 558 <= title_number <= 624:
                label = "Kneipe"
            else:
                label = "Allerhand Humor"
            
            labeled_songs.append((label, title, text))
    
    return labeled_songs

# Start- und Endseite, jedoch ohne Inhaltsverzeichniss, Anhang, etc. So wird sichergestellt, dass nur Liedertexte und keine andere Informationen extrahiert werden
start_page = 1
end_page = 363

# Extrahieren der Lieder
songs = extract_song_titles_and_texts(start_page, end_page)

# Debugging: Anzeigen der Titel vor der Filterung. Da die HTML-Seiten doch ein wenig uneinheitlich waren, wurden fälschlicherweise andere Einträge (Ziffern in Eckigen Klammern, 'Titel: [Ziffer]' und 'Herunterladen') als Titel erkannt, jedoch nur in Fällen, wo die Liedtexte denselben Lieder auf mehreren HTML-Seiten zu finden waren. Dies wurde umgegangen, indem derartige Titel rausgefiltert wurden und die Liedtexte nach diesen zu dem vorherigen, korrekten Titel hinzugefügt
#print('Vor der Filterung:')
#for title, text in songs:
#    print(f'Titel: '{title}'')

# Titel, die nur aus Ziffern in eckigen Klammern bestehen oder im Format "Titel: [Ziffer]" sind, werden gefiltert
filtered_songs = []
for title, text in songs:
    if not re.match(r'^\[\d+\]$', title) and not re.match(r'^Titel:\s*\[\d+\]$', title) and "Herunterladen" not in title:
        filtered_songs.append((title, text))
    else:
        # Der Text des gefilterten Titels wird zum vorherigen Text hinzugefügt
        if filtered_songs:
            filtered_songs[-1] = (filtered_songs[-1][0], filtered_songs[-1][1] + " " + text)

# Labeln der Lieder
labeled_songs = label_songs(filtered_songs)

# Die Ergebnisse werden in einer TXT-Datei 'kommersbuch_output.txt' gespeichert
with open("kommersbuch_output.txt", "w", encoding="utf-8") as file:
    for label, title, text in labeled_songs:
        file.write(f'Label: {label}\nTitel: {title}\nText:\n{text}\n\n')