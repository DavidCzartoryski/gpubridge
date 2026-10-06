#!/bin/sh
# Download a wheel into a cache directory, resuming partial downloads, and
# check it against the sha256 published in the PyTorch index. Prints the path.
#
#   fetch_wheel.sh URL SHA256 DIR
set -eu
url=$1 sha=$2 dir=$3
mkdir -p "$dir"
dest="$dir/$(basename "$url" | sed 's/%2B/+/g')"

for attempt in 1 2 3; do
    if [ -f "$dest" ] && echo "$sha  $dest" | sha256sum -c --status; then
        echo "$dest"
        exit 0
    fi
    curl -fsSL --retry 10 --retry-delay 5 --retry-all-errors -C - -o "$dest" "$url" || true
    if echo "$sha  $dest" | sha256sum -c --status; then
        echo "$dest"
        exit 0
    fi
    echo "fetch_wheel: checksum mismatch on attempt $attempt, starting over" >&2
    rm -f "$dest"
done
echo "fetch_wheel: could not download a valid $url" >&2
exit 1
