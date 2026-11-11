#!/bin/bash
# Installs "Vorlesung" as a Mac app (Apple Silicon recommended).
#
#   ./install_mac.sh
#
# What it does:
#   1. creates a private Python environment in ~/Library/Application Support/Vorlesung
#   2. installs the app and its dependencies (Whisper/MLX, pywebview, …)
#   3. builds Vorlesung.app in /Applications (or ~/Applications)
#   4. downloads the speech model once (~1.6 GB), so the first lecture starts instantly
#
# Run it again at any time to update (e.g. after `git pull`).

set -euo pipefail

APP_NAME="Vorlesung"
BUNDLE_ID="de.vorlesung.transkript"
SUPPORT_DIR="$HOME/Library/Application Support/$APP_NAME"
VENV="$SUPPORT_DIR/venv"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bold() { printf "\033[1m%s\033[0m\n" "$*"; }
fail() { printf "\033[31mFehler:\033[0m %s\n" "$*" >&2; exit 1; }

[[ "$(uname)" == "Darwin" ]] || fail "Dieses Skript ist für macOS."
if [[ "$(uname -m)" != "arm64" ]]; then
  echo "Hinweis: Intel-Mac erkannt – die Erkennung läuft auf der CPU (kleineres Modell, langsamer)."
fi

# --- 1. Python ---------------------------------------------------------------
bold "1/4  Suche Python (3.10 oder neuer)…"
PYTHON=""
for cand in python3.13 python3.12 python3.11 python3.10 \
            /opt/homebrew/bin/python3 /usr/local/bin/python3 python3; do
  if command -v "$cand" >/dev/null 2>&1; then
    if "$cand" -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)' 2>/dev/null; then
      PYTHON="$(command -v "$cand")"
      break
    fi
  fi
done
if [[ -z "$PYTHON" ]]; then
  echo
  echo "Kein passendes Python gefunden. Am einfachsten über Homebrew (https://brew.sh):"
  echo "    brew install python@3.12"
  echo "oder den Installer von https://www.python.org/downloads/macos/ verwenden."
  echo "Danach dieses Skript erneut starten."
  exit 1
fi
echo "     $PYTHON ($("$PYTHON" --version))"

# --- 2. Environment + packages ----------------------------------------------
bold "2/4  Installiere die App (kann ein paar Minuten dauern)…"
mkdir -p "$SUPPORT_DIR"
if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --quiet --upgrade pip
"$VENV/bin/python" -m pip install --quiet --upgrade "$REPO_DIR"
# Make sure the newest code is installed even if the version number didn't change.
"$VENV/bin/python" -m pip install --quiet --force-reinstall --no-deps "$REPO_DIR"

# --- 3. App bundle ------------------------------------------------------------
bold "3/4  Erstelle $APP_NAME.app…"
if [[ -w "/Applications" ]]; then
  APPS_DIR="/Applications"
else
  APPS_DIR="$HOME/Applications"
  mkdir -p "$APPS_DIR"
fi
APP="$APPS_DIR/$APP_NAME.app"
rm -rf "$APP"
mkdir -p "$APP/Contents/MacOS" "$APP/Contents/Resources"

cat > "$APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key><string>$APP_NAME</string>
  <key>CFBundleDisplayName</key><string>$APP_NAME</string>
  <key>CFBundleIdentifier</key><string>$BUNDLE_ID</string>
  <key>CFBundleVersion</key><string>1</string>
  <key>CFBundleShortVersionString</key><string>0.1</string>
  <key>CFBundlePackageType</key><string>APPL</string>
  <key>CFBundleExecutable</key><string>$APP_NAME</string>
  <key>CFBundleIconFile</key><string>AppIcon</string>
  <key>LSMinimumSystemVersion</key><string>12.0</string>
  <key>NSHighResolutionCapable</key><true/>
  <key>NSMicrophoneUsageDescription</key>
  <string>Vorlesung nimmt über das Mikrofon auf, um die Vorlesung lokal auf deinem Mac zu transkribieren.</string>
</dict>
</plist>
PLIST

cat > "$APP/Contents/MacOS/$APP_NAME" <<LAUNCHER
#!/bin/bash
# Starts the Vorlesung app from its private Python environment.
LOG="\$HOME/Library/Logs/$APP_NAME.log"
export VORLESUNG_ICON="\$(cd "\$(dirname "\$0")/../Resources" && pwd)/AppIcon.icns"
exec "$VENV/bin/python" -m lecture_transcriber.app "\$@" >>"\$LOG" 2>&1
LAUNCHER
chmod +x "$APP/Contents/MacOS/$APP_NAME"

# Icon
ICONSET="$(mktemp -d)/AppIcon.iconset"
mkdir -p "$ICONSET"
"$VENV/bin/python" -m lecture_transcriber.icon "$ICONSET/icon_512x512@2x.png" 1024
for size in 16 32 128 256 512; do
  sips -z $size $size "$ICONSET/icon_512x512@2x.png" --out "$ICONSET/icon_${size}x${size}.png" >/dev/null
  double=$((size * 2))
  sips -z $double $double "$ICONSET/icon_512x512@2x.png" --out "$ICONSET/icon_${size}x${size}@2x.png" >/dev/null
done
iconutil -c icns "$ICONSET" -o "$APP/Contents/Resources/AppIcon.icns" || echo "     (Icon konnte nicht erstellt werden – egal)"

# Ad-hoc signature, so macOS remembers the microphone permission.
codesign --force --deep --sign - "$APP" >/dev/null 2>&1 || true
touch "$APP"
# Register with Launch Services and Spotlight so the app shows up in search right away.
LSREGISTER="/System/Library/Frameworks/CoreServices.framework/Frameworks/LaunchServices.framework/Support/lsregister"
[[ -x "$LSREGISTER" ]] && "$LSREGISTER" -f "$APP" >/dev/null 2>&1 || true
mdimport "$APP" >/dev/null 2>&1 || true
echo "     $APP"

# --- 4. Model -----------------------------------------------------------------
bold "4/4  Lade das Spracherkennungsmodell (einmalig, ~1,6 GB)…"
"$VENV/bin/python" - <<'PY'
from lecture_transcriber.transcriber import Transcriber
t = Transcriber()
print("     " + t.description)
PY

echo
bold "Fertig! 🎓"
echo "Starte „${APP_NAME}“ aus dem Programme-Ordner, über Spotlight (⌘ Leertaste) oder mit:"
echo "    open \"$APP\""
echo "Beim ersten Start fragt macOS nach dem Mikrofon-Zugriff – bitte erlauben."
echo "Transkripte landen in ~/Vorlesungen."
