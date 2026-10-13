#!/usr/bin/env bash
# Smoke-test an installed syncrec package:
#
#   sudo apt install ./dist/syncrec_*.deb xvfb
#   packaging/linux/test_package.sh
#
# Checks the command line, a short recording with dummy cameras (encoding with the
# system's ffmpeg), pypylon (the pylon camera emulator must be found), and that the
# GUI starts.
set -euo pipefail

WORK="$(mktemp -d)"
cd "$WORK"
cat > test.json <<'EOF'
{
  "acquisition": {"backend": "dummy", "fps": 10, "roi": {"width": 320, "height": 240}},
  "dummy": {"count": 2},
  "sync": {"mode": "software_trigger"},
  "session": {"output_dir": "rec", "clip_name": "pkgtest"},
  "encoder": {"codec": "h264", "hardware": "cpu"}
}
EOF

echo "== command line"
syncrec --help > /dev/null

echo "== settings are per user"
syncrec config --write "$HOME/.config/syncrec/settings.json" > /dev/null
python3 - <<'PY'
import os, subprocess
out = subprocess.run(["syncrec", "-v", "list", "--backend", "dummy"],
                     capture_output=True, text=True).stderr
expected = os.path.expanduser("~/.config/syncrec/settings.json")
assert f"loaded config from {expected}" in out, out[-2000:]
PY
rm -f "$HOME/.config/syncrec/settings.json"

echo "== recording with dummy cameras"
syncrec record -c test.json --duration 3
videos=$(find rec -name '*.mp4' | wc -l)
[ "$videos" -eq 2 ] || { echo "FAIL: expected 2 videos, found $videos"; exit 1; }

echo "== pypylon (camera emulator)"
PYLON_CAMEMU=2 syncrec list -c test.json --backend pylon > list.txt 2>&1 || true
cat list.txt
grep -q "cameras: 2" list.txt || { echo "FAIL: pylon emulator not found"; exit 1; }

echo "== GUI"
# Still running when the timeout ends it (exit 124) = it started; anything else is
# a crash, typically a missing system library for Qt.
set +e
QT_QPA_PLATFORM=xcb xvfb-run -a timeout 20 syncrec gui -c test.json
code=$?
set -e
[ "$code" -eq 124 ] || { echo "FAIL: the GUI exited with code $code"; exit 1; }

rm -rf "$WORK"
echo "PACKAGE TEST PASSED"
