#!/usr/bin/env bash
# Server regression gate: parity with the CLI (idle server == CLI bytes), multi-doc concurrency,
# heterogeneous windows (base+gundam co-batch: agreement, no head-of-line blocking, mixed tilings),
# and error paths. Run after touching engine.cu/server.cpp/vision_enc.cu:
#   make servercheck        (builds KV=fp8 first, same config as tools/check.sh)
# Needs the GPU + weights; takes ~1-2 min (one weight load, one server process).
set -uo pipefail
cd "$(dirname "$0")/.."
PDF="${1:-$HOME/unlimited-ocr/Unlimited-OCR.pdf}"
T=$(mktemp -d); trap 'kill $SPID 2>/dev/null; rm -rf $T' EXIT
rc=0
fail(){ echo "  FAIL: $1"; rc=1; }

# GSLOTS=8 for the TEST server: the gate spawns server + CLI-comparison runs on one GPU (often next to a
# production instance); the default 32 big slots (+3.7GB/server) can OOM the CLI run -> empty-file "parity
# failures". 8 slots exercise the same heterogeneous-window code paths at CI footprint.
GSLOTS="${GSLOTS_CHECK:-8}" ./ocr_bin serve 0 > $T/serve.log 2>&1 & SPID=$!
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

echo "== base parity preserved after a gundam job ran (vision-input invalidation) =="
curl -sS --data-binary @"$PDF" "$U/ocr?pages=1" -o $T/srv2.txt
{ cat $T/srv2.txt; echo; } | diff - $T/cli.txt >/dev/null && echo "  OK" || fail "post-gundam parity"

echo "== concurrent documents (3 in flight, one arrives late) =="
curl -sS --data-binary @"$PDF" "$U/ocr?pages=4" -o /dev/null -w '%{http_code}\n' > $T/r1 & P1=$!
curl -sS --data-binary @"$PDF" "$U/ocr?pages=4" -o /dev/null -w '%{http_code}\n' > $T/r2 & P2=$!
( sleep 1; curl -sS --data-binary @"$PDF" "$U/ocr?pages=1" -o /dev/null -w '%{http_code}\n' > $T/r3 ) & P3=$!
wait $P1 $P2 $P3    # NOT bare `wait` — that would also wait on the backgrounded server
[ "$(cat $T/r1)$(cat $T/r2)$(cat $T/r3)" = 200200200 ] && echo "  OK" || fail "concurrent docs: $(cat $T/r1 $T/r2 $T/r3 | tr '\n' ' ')"

echo "== selective page list =="
curl -sS --data-binary @"$PDF" "$U/ocr?pages=2,1" -D $T/pl.h -o /dev/null || fail "pages list POST"
grep -q "X-Pages: 2" $T/pl.h || fail "pages=2,1 -> X-Pages: 2"
grep -q "X-Page-Conf:" $T/pl.h || fail "X-Page-Conf header missing"
curl -sS --data-binary @"$PDF" "$U/ocr?pages=2%2C1" -D $T/pe.h -o /dev/null || fail "percent-encoded list POST"
grep -q "X-Pages: 2" $T/pe.h || fail "pages=2%2C1 (URLSearchParams comma) -> X-Pages: 2"
[ "$(curl -sS -o /dev/null -w '%{http_code}' -X POST "$U/ocr?pages=999,1" --data-binary @"$PDF")" = 400 ] || fail "out-of-range page -> 400"
[ $rc -eq 0 ] && echo "  OK"

echo "== large-doc response headers (per-page conf; regression: resp() buffer overflow) =="
BIG="$HOME/unlimited-ocr/testdata/paper_x8.pdf"   # 112 pages -> ~1.1KB of X-Page-Conf/LowConf headers
if [ -f "$BIG" ]; then
  curl -sS --max-time 120 --data-binary @"$BIG" "$U/ocr" -D $T/big.h -o $T/big.txt || fail "big-doc POST"
  grep -q "X-Pages: 112" $T/big.h || fail "big-doc X-Pages"
  nc=$(grep -i x-page-conf: $T/big.h | tr ',' '\n' | wc -l)
  [ "$nc" = 112 ] || fail "X-Page-Conf has $nc values (want 112)"
  python3 -c "import sys; sys.exit(1 if open('$T/big.h','rb').read().count(b'\0') else 0)" || fail "NUL bytes in response headers (resp() over-read)"
  [ $rc -eq 0 ] && echo "  OK (112-page headers clean)"
else echo "  SKIP (no paper_x8.pdf)"; fi

echo "== server-side auto hi-res (base job re-OCRs low-conf pages in gundam before returning) =="
BROCH="$HOME/unlimited-ocr/testdata/reaktor_mkt.pdf"   # small-text brochure: base flags pages 1-2, server upgrades them
if [ -f "$BROCH" ]; then
  ca=$(curl -sS --max-time 120 --data-binary @"$BROCH" "$U/ocr?pages=4&auto=0" -D - -o /dev/null | grep -i x-page-conf: | grep -oE '[0-9.]+' | head -1)
  cb=$(curl -sS --max-time 120 --data-binary @"$BROCH" "$U/ocr?pages=4"        -D - -o /dev/null | grep -i x-page-conf: | grep -oE '[0-9.]+' | head -1)
  echo "  page1 conf: auto=off $ca -> auto=on(default) $cb"
  awk "BEGIN{exit !($ca<0.80 && $cb>$ca+0.08)}" && echo "  OK (server auto-upgraded flagged page)" || fail "auto hi-res did not upgrade a flagged page ($ca -> $cb)"
else echo "  SKIP (no reaktor_mkt.pdf)"; fi

echo "== gundam windowing (>old 64-page cap must NOT 413; VRAM bounded by window) =="
if [ -f "$BIG" ]; then   # paper_x8 = 112 pages; gundam-decode 68 (over the removed cap) -> 200, not 413
  code=$(curl -sS --max-time 600 --data-binary @"$BIG" "$U/ocr?pages=68&gundam=1" -o $T/gwin.txt -D $T/gwin.h -w '%{http_code}')
  [ "$code" = 200 ] || fail "gundam 68pg -> $code (expected 200; windowing removed the page cap)"
  grep -q "X-Pages: 68" $T/gwin.h || fail "gundam 68pg X-Pages"
  [ $rc -eq 0 ] && echo "  OK (68 gundam pages, no cap)"
else echo "  SKIP (no paper_x8.pdf)"; fi

echo "== heterogeneous window: base + gundam co-batch (agreement vs idle, both 200) =="
if [ -f "$BROCH" ]; then
  curl -sS --max-time 120 --data-binary @"$PDF" "$U/ocr?pages=1&auto=0" -o $T/het_idle.txt || fail "idle ref POST"
  curl -sS --max-time 300 --data-binary @"$BROCH" "$U/ocr?pages=8&gundam=1" -o /dev/null -w '%{http_code}' > $T/het_g & PG=$!
  sleep 1
  curl -sS --max-time 300 --data-binary @"$PDF" "$U/ocr?pages=1&auto=0" -o $T/het_base.txt -w '%{http_code}' > $T/het_b
  wait $PG
  [ "$(cat $T/het_b)$(cat $T/het_g)" = 200200 ] || fail "co-batch statuses: base=$(cat $T/het_b) gundam=$(cat $T/het_g)"
  ag=$(python3 -c "import difflib;a=open('$T/het_idle.txt').read();b=open('$T/het_base.txt').read();print(f'{difflib.SequenceMatcher(None,a,b).ratio():.4f}')")
  echo "  base agreement idle-vs-co-batched: $ag"
  awk "BEGIN{exit !($ag>=0.99)}" && echo "  OK (>=0.99 agreement, near-tie class)" || fail "co-batch agreement $ag < 0.99"
else echo "  SKIP (no reaktor_mkt.pdf)"; fi

echo "== no head-of-line blocking: 1pg base during 50pg gundam finishes early =="
if [ -f "$BROCH" ]; then
  curl -sS --max-time 600 --data-binary @"$BROCH" "$U/ocr?gundam=1" -o /dev/null -w '%{time_total}' > $T/hol_g & PG=$!
  sleep 2
  bt=$(curl -sS --max-time 600 --data-binary @"$PDF" "$U/ocr?pages=1" -o /dev/null -w '%{time_total}')
  wait $PG; gt=$(cat $T/hol_g)
  echo "  base 1pg: ${bt}s while gundam 50pg ran ${gt}s"
  awk "BEGIN{exit !($bt < 15 && $bt+2 < $gt)}" && echo "  OK (base returned well before the gundam job)" || fail "base 1pg took ${bt}s during gundam job (${gt}s) — HOL blocking?"
else echo "  SKIP (no reaktor_mkt.pdf)"; fi

echo "== mixed-tiling gundam doc decodes in one heterogeneous pass =="
MIX="$HOME/unlimited-ocr/testdata/mixed_ratio.pdf"   # 3 pages, 3 different gundam tilings (was: sequential fallback)
if [ -f "$MIX" ]; then
  code=$(curl -sS --max-time 120 --data-binary @"$MIX" "$U/ocr?gundam=1" -o $T/mix.txt -D $T/mix.h -w '%{http_code}')
  [ "$code" = 200 ] || fail "mixed-tiling gundam -> $code"
  grep -q "X-Pages: 3" $T/mix.h || fail "mixed-tiling X-Pages"
  [ $(wc -c < $T/mix.txt) -gt 1000 ] || fail "mixed-tiling output suspiciously small"
  [ $rc -eq 0 ] && echo "  OK (3 tilings, one job)"
else echo "  SKIP (no mixed_ratio.pdf)"; fi

echo "== annotation viewer statics =="
curl -sS $U/ | grep -q "annotation viewer" || fail "GET / viewer html"
[ "$(curl -sS -o /dev/null -w '%{http_code}' $U/pdf.mjs)" = 200 ] || fail "GET /pdf.mjs"
[ "$(curl -sS -o /dev/null -w '%{http_code}' $U/pdf.worker.mjs)" = 200 ] || fail "GET /pdf.worker.mjs"
[ $rc -eq 0 ] && echo "  OK"

echo "== error paths =="
head -c 4096 /dev/urandom > $T/junk.bin
[ "$(curl -sS --data-binary @$T/junk.bin $U/ocr -o /dev/null -w '%{http_code}')" = 422 ] || fail "garbage -> 422"
[ "$(curl -sS -X POST $U/ocr -H 'Transfer-Encoding: chunked' --data-binary @$T/junk.bin -o /dev/null -w '%{http_code}')" = 501 ] || fail "chunked -> 501"
[ "$(curl -sS -o /dev/null -w '%{http_code}' $U/nope)" = 404 ] || fail "404"
[ "$(curl -sS -o /dev/null -w '%{http_code}' -X POST "$U/ocr?pages=0" --data-binary @$T/junk.bin)" = 400 ] || fail "pages=0 -> 400"
[ $rc -eq 0 ] && echo "  OK"

[ $rc -eq 0 ] && echo "SERVER CHECK PASSED" || echo "SERVER CHECK FAILED"
exit $rc
