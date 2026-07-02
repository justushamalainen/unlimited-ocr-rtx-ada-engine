#!/usr/bin/env bash
# Server regression gate: parity with the CLI (idle server == CLI bytes), multi-doc concurrency,
# gundam interlude + re-entry, and error paths. Run after touching engine.cu/server.cpp/vision_enc.cu:
#   make servercheck        (builds KV=fp8 first, same config as tools/check.sh)
# Needs the GPU + weights; takes ~1-2 min (one weight load, one server process).
set -uo pipefail
cd "$(dirname "$0")/.."
PDF="${1:-$HOME/unlimited-ocr/Unlimited-OCR.pdf}"
T=$(mktemp -d); trap 'kill $SPID 2>/dev/null; rm -rf $T' EXIT
rc=0
fail(){ echo "  FAIL: $1"; rc=1; }

./ocr_bin serve 0 > $T/serve.log 2>&1 & SPID=$!
for i in $(seq 120); do
  PORT=$(grep -oE 'ready on port [0-9]+' $T/serve.log | grep -oE '[0-9]+$' || true); [ -n "$PORT" ] && break
  kill -0 $SPID 2>/dev/null || { echo "server died:"; tail -5 $T/serve.log; exit 1; }
  sleep 1
done
[ -n "${PORT:-}" ] || { echo "server never became ready"; exit 1; }
U="http://127.0.0.1:$PORT"

echo "== healthz =="
[ "$(curl -sS -o /dev/null -w '%{http_code}' $U/healthz)" = 200 ] && echo "  OK" || fail "healthz"

echo "== idle parity: server body == CLI text (base 1pg) =="
curl -sS --data-binary @"$PDF" "$U/ocr?pages=1" -o $T/srv.txt || fail "POST"
./ocr_bin "$PDF" 1 60 2>/dev/null | sed -n '/===== OCR/,$p' | tail -n +2 > $T/cli.txt
{ cat $T/srv.txt; echo; } | diff - $T/cli.txt >/dev/null && echo "  OK" || fail "base parity"

echo "== idle parity: gundam 1pg =="
curl -sS --data-binary @"$PDF" "$U/ocr?pages=1&gundam=1" -o $T/srvg.txt || fail "POST gundam"
GUNDAM=1 ./ocr_bin "$PDF" 1 60 2>/dev/null | sed -n '/===== OCR/,$p' | tail -n +2 > $T/clig.txt
{ cat $T/srvg.txt; echo; } | diff - $T/clig.txt >/dev/null && echo "  OK" || fail "gundam parity"

echo "== base re-entry after gundam interlude =="
curl -sS --data-binary @"$PDF" "$U/ocr?pages=1" -o $T/srv2.txt
{ cat $T/srv2.txt; echo; } | diff - $T/cli.txt >/dev/null && echo "  OK" || fail "re-entry parity"

echo "== concurrent documents (3 in flight, one arrives late) =="
curl -sS --data-binary @"$PDF" "$U/ocr?pages=4" -o /dev/null -w '%{http_code}\n' > $T/r1 & P1=$!
curl -sS --data-binary @"$PDF" "$U/ocr?pages=4" -o /dev/null -w '%{http_code}\n' > $T/r2 & P2=$!
( sleep 1; curl -sS --data-binary @"$PDF" "$U/ocr?pages=1" -o /dev/null -w '%{http_code}\n' > $T/r3 ) & P3=$!
wait $P1 $P2 $P3    # NOT bare `wait` — that would also wait on the backgrounded server
[ "$(cat $T/r1)$(cat $T/r2)$(cat $T/r3)" = 200200200 ] && echo "  OK" || fail "concurrent docs: $(cat $T/r1 $T/r2 $T/r3 | tr '\n' ' ')"

echo "== error paths =="
head -c 4096 /dev/urandom > $T/junk.bin
[ "$(curl -sS --data-binary @$T/junk.bin $U/ocr -o /dev/null -w '%{http_code}')" = 422 ] || fail "garbage -> 422"
[ "$(curl -sS -X POST $U/ocr -H 'Transfer-Encoding: chunked' --data-binary @$T/junk.bin -o /dev/null -w '%{http_code}')" = 501 ] || fail "chunked -> 501"
[ "$(curl -sS -o /dev/null -w '%{http_code}' $U/nope)" = 404 ] || fail "404"
[ "$(curl -sS -o /dev/null -w '%{http_code}' -X POST "$U/ocr?pages=0" --data-binary @$T/junk.bin)" = 400 ] || fail "pages=0 -> 400"
[ $rc -eq 0 ] && echo "  OK"

[ $rc -eq 0 ] && echo "SERVER CHECK PASSED" || echo "SERVER CHECK FAILED"
exit $rc
