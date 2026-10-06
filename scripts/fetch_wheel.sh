#!/bin/sh
# Download a wheel into a directory, resuming partial downloads, and check it
# against the sha256 published in the PyTorch index. Prints the path.
# Runs on Linux (sha256sum) or macOS (shasum).
#
#   fetch_wheel.sh URL SHA256 DIR
set -eu
url=$1 sha=$2 dir=$3
mkdir -p "$dir"
dest="$dir/$(basename "$url" | sed 's/%2B/+/g')"

# Compare digests as text: macOS now ships a BSD sha256sum without --status,
# so prefer shasum (macOS, most Linux) and fall back to GNU sha256sum.
digest() {
    if command -v shasum >/dev/null 2>&1; then
        shasum -a 256 "$1" | cut -d' ' -f1
    else
        sha256sum "$1" | cut -d' ' -f1
    fi
}

sha_ok() {
    [ -f "$dest" ] && [ "$(digest "$dest")" = "$sha" ]
}

for attempt in 1 2 3; do
    if sha_ok; then
        echo "$dest"
        exit 0
    fi
    curl -fsSL --retry 10 --retry-delay 5 --retry-all-errors -C - -o "$dest" "$url" || true
    if sha_ok; then
        echo "$dest"
        exit 0
    fi
    echo "fetch_wheel: checksum mismatch on attempt $attempt, starting over" >&2
    rm -f "$dest"
done
echo "fetch_wheel: could not download a valid $url" >&2
exit 1
