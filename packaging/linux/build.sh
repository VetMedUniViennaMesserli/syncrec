#!/usr/bin/env bash
# Build the Linux installer (.deb).
#
#   packaging/linux/build.sh 1.2.0
#
# Needs only dpkg-deb. Output: dist/syncrec_<version>_all.deb
#
# The package contains only this repository's code. Python, ffmpeg and the system
# libraries Qt needs come from the distribution (package dependencies); the Python
# packages (pypylon, NumPy, PySide6) are downloaded from PyPI by the package's
# postinst script during installation, into /opt/syncrec/venv. Nothing from other
# projects is redistributed.
set -euo pipefail

VERSION="${1:?usage: build.sh VERSION}"
VERSION="${VERSION#v}"
MAINTAINER="${MAINTAINER:-syncrec maintainers <noreply@github.com>}"
HOMEPAGE="${HOMEPAGE:-}"

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
HERE="$ROOT/packaging/linux"
PKG="$ROOT/build/linux/deb"
DIST="$ROOT/dist"
rm -rf "$PKG"
mkdir -p "$PKG/DEBIAN" "$PKG/opt/syncrec/app" "$PKG/usr/bin" \
         "$PKG/usr/share/applications" "$PKG/usr/share/doc/syncrec" "$DIST"

# The program: the syncrec package, the tools (`syncrec check` uses them) and the
# files it reads at run time.
cd "$ROOT"
tar --exclude=__pycache__ --exclude='*.pyc' -cf - syncrec tools requirements.txt \
    config.example.json | tar -C "$PKG/opt/syncrec/app" -xf -
sed -i "s/^__version__ = .*/__version__ = \"$VERSION\"/" \
    "$PKG/opt/syncrec/app/syncrec/__init__.py"

install -m 755 "$HERE/syncrec" "$PKG/usr/bin/syncrec"
install -m 644 "$HERE/syncrec.desktop" "$PKG/usr/share/applications/"
install -m 644 README.md "$PKG/usr/share/doc/syncrec/"
install -m 644 LICENSE "$PKG/usr/share/doc/syncrec/copyright"
install -m 755 "$HERE/postinst" "$HERE/postrm" "$PKG/DEBIAN/"

sed -e "s|@VERSION@|$VERSION|" \
    -e "s|@MAINTAINER@|$MAINTAINER|" \
    -e "s|@SIZE@|$(du -sk "$PKG/opt" | cut -f1)|" \
    "$HERE/control.in" > "$PKG/DEBIAN/control"
if [ -n "$HOMEPAGE" ]; then
    sed -i "/^Section:/i Homepage: $HOMEPAGE" "$PKG/DEBIAN/control"
fi

dpkg-deb --build --root-owner-group "$PKG" "$DIST/syncrec_${VERSION}_all.deb"
ls -lh "$DIST"
