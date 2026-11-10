# 🎓 Vorlesung – Live-Transkription ohne Zeitlimit

Eine Mac-App, die Vorlesungen **live mitschreibt**: ohne Abo, ohne 30-Minuten-Grenze
und ohne Cloud. Die Spracherkennung (OpenAI Whisper `large-v3-turbo`) läuft komplett
lokal auf deinem MacBook. Auf Apple-Silicon-Macs (M1–M4) nutzt sie dafür die GPU (MLX).

- **Live-Text** während der Vorlesung, mit Zeitmarken
- **Kein Zeitlimit**: läuft so lange, wie du willst
- **Speichern unter…** als Word (.docx), Text, Markdown, Untertitel (.srt) oder Audio (.wav)
- **Archiv** aller Vorlesungen in der Seitenleiste
- **Aufnahmen & Videos** (MP3, M4A, MP4, MOV …) per Drag & Drop nachträglich transkribieren
- **Fachbegriffe** angeben (z.B. „Eigenwert, Determinante, Prof. Müller“) für die richtige Schreibweise
- **Absturzsicher**: Text und Audio werden laufend gespeichert, der Mac bleibt während der Aufnahme wach
- Die Audioaufnahme bleibt als Backup erhalten
- **Hell/Dunkel** per Knopf oben rechts (ohne Auswahl folgt die App dem macOS-Design)

## Installation (einmalig, ca. 5 Minuten)

Voraussetzung: Python 3.10 oder neuer. Falls nicht vorhanden, am einfachsten mit
[Homebrew](https://brew.sh):

```bash
brew install python@3.12
```

Dann im Terminal:

```bash
git clone https://github.com/luca22303/Transcription.git
```

```bash
cd Transcription && ./install_mac.sh
```

Das Skript richtet eine eigene Python-Umgebung ein, erstellt **Vorlesung.app** im
Programme-Ordner und lädt einmalig das Sprachmodell (~1,6 GB). Danach startest du die
App wie jede andere (Spotlight: ⌘ Leertaste → „Vorlesung“).

Beim ersten Start fragt macOS nach dem **Mikrofon-Zugriff**. Bitte erlauben.
(Später änderbar unter Systemeinstellungen › Datenschutz & Sicherheit › Mikrofon.)

**Aktualisieren:** `git pull && ./install_mac.sh`

## Benutzung

1. **Neue Aufnahme** → Name der Vorlesung eingeben (z.B. „Analysis I“)
2. Optional: Fachbegriffe eintragen. Sie werden pro Vorlesung gemerkt.
3. **Aufnahme starten**. Der Text erscheint mit wenigen Sekunden Verzögerung.
4. **Stopp** → **Speichern unter…** → Format wählen

Alle Vorlesungen landen außerdem automatisch in `~/Vorlesungen/`:

```
~/Vorlesungen/2026-11-10_10-15_Analysis-I/
├── transkript.md     ← mit Zeitmarken
├── transkript.txt    ← reiner Text (gut für ChatGPT/Claude zum Zusammenfassen)
├── transkript.srt    ← Untertitel
├── aufnahme.wav      ← Audio-Backup (~115 MB pro Stunde)
└── info.json
```

### Tipps für den Hörsaal

- Möglichst **weit vorne** sitzen. Das eingebaute MacBook-Mikrofon reicht dann gut aus.
- Den **Deckel offen** lassen. Beim Zuklappen schaltet macOS das Mikrofon ab.
- Am Netzteil oder mit vollem Akku: Die Erkennung braucht etwas Strom (bei M-Chips wenig).

### Online-Vorlesungen (Zoom, Teams, Panopto-Livestream)

Um den **Systemton** statt des Mikrofons aufzunehmen, brauchst du den kostenlosen
virtuellen Audiotreiber [BlackHole](https://existential.audio/blackhole/):

```bash
brew install blackhole-2ch
```

1. Programm **Audio-MIDI-Setup** öffnen → unten links „+“ → **Gerät mit mehreren Ausgängen erstellen**
2. Dort **BlackHole 2ch** und deine **Lautsprecher/Kopfhörer** anhaken
3. In den Systemeinstellungen › Ton › Ausgabe dieses neue Gerät wählen (du hörst weiterhin alles)
4. In der Vorlesung-App als Audioquelle **BlackHole 2ch** auswählen

Nach der Vorlesung die Tonausgabe wieder auf die Lautsprecher zurückstellen.

### Aufzeichnungen & Videos

Datei auf das Feld „Aufnahme oder Video hier ablegen“ ziehen. Das Transkript erscheint
live und landet ebenfalls im Archiv. Unterstützt werden alle gängigen Formate
(MP3, M4A, WAV, MP4, MOV, MKV, WebM …).

## Terminal-Befehle (optional)

Nach der Installation gibt es auch einen Befehl `vorlesung`
(`~/Library/Application Support/Vorlesung/venv/bin/vorlesung`):

| Befehl | Was es tut |
|---|---|
| `vorlesung` | App öffnen |
| `vorlesung live "Analysis I"` | Im Terminal aufnehmen, Strg+C zum Beenden |
| `vorlesung datei aufnahme.m4a` | Datei transkribieren |
| `vorlesung geraete` | Mikrofone/Eingänge auflisten |

Nützliche Optionen: `-m small` (kleineres, schnelleres Modell), `-l en` (englische Vorlesung),
`-p "Fachbegriffe"`, `-d BlackHole` (Audioquelle), `-o Ordner`.

## Wie es funktioniert

```
Mikrofon ──► Aufnahme-Thread ──► aufnahme.wav (laufend gespeichert)
                  │
                  ▼
        Schnitt an Sprechpausen (6–25 s)
                  │
                  ▼
        Silero-VAD: Stille überspringen
                  │
                  ▼
   Whisper large-v3-turbo (MLX auf der Apple-GPU)
   + Kontext der letzten Sätze + deine Fachbegriffe
                  │
                  ▼
   Filter gegen typische Whisper-„Halluzinationen“
   („Untertitel im Auftrag des ZDF“ & Co.)
                  │
                  ▼
   transkript.md/.txt/.srt  +  Live-Ansicht in der App
```

Aufnahme und Erkennung laufen getrennt. Falls die Erkennung kurz hinterherhinkt, geht
trotzdem kein Ton verloren.

| Mac | Backend | Modell (Standard) |
|---|---|---|
| Apple Silicon (M1–M4) | MLX (GPU) | `large-v3-turbo` |
| Intel-Mac | faster-whisper (CPU) | `small` |

Die App läuft nur lokal (`127.0.0.1`). Nichts wird hochgeladen, und andere Webseiten
können sie nicht fernsteuern.

## Probleme?

- **„Kein Ton vom Mikrofon“**: Mikrofon-Zugriff für „Vorlesung“ erlauben (Systemeinstellungen ›
  Datenschutz & Sicherheit › Mikrofon), dann die App neu starten.
- **App startet nicht**: Log ansehen mit `open ~/Library/Logs/Vorlesung.log`
- **Text hinkt hinterher**: anderes rechenintensives Programm schließen, oder im Terminal mit
  einem kleineren Modell starten: `vorlesung live -m medium`
- **„App ist beschädigt“ / nicht verifiziert**: Rechtsklick auf die App → Öffnen

## Entwicklung

```bash
python3 -m venv .venv && source .venv/bin/activate
```

```bash
pip install -e . pytest && pytest
```

`vorlesung app --browser` öffnet die Oberfläche im Browser statt im App-Fenster.
