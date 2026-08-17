#!/usr/bin/env bash
# ============================================================================
# Ternary Bonsai 8B - output legibility gate (OPTIMIZATION_PLAN Step 6).
#
# MANDATORY before recording any speed number for a new build: send a handful
# of real prompts through the server and confirm each reply is legible,
# coherent text - not garbage, not empty. A fast kernel that produces garbage
# is not a candidate, full stop.
#
#   bash verify-output.sh [--model PATH] [--prompts N] [--port P] [--ctx N]
#
# Env: LLAMA_SERVER=/path/to/llama-server  (default: bt build/models resolution)
# Exits 0 only if every sampled reply is non-empty, printable and coherent.
# ============================================================================

set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MODEL_PATH="standard/Ternary-Bonsai-8B-Q2_0_g64.gguf"
PROMPTS=5
PORT=8095
CTX=4096
NGL=999
LLAMA_SERVER="${LLAMA_SERVER:-}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model) MODEL_PATH="$2"; shift 2 ;;
        --prompts) PROMPTS="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --ctx) CTX="$2"; shift 2 ;;
        -h|--help) sed -n 's/^# \{0,1\}//p' "${BASH_SOURCE[0]}" | head -n 14; exit 0 ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
done

sibling_roots() {
    local p="$ROOT" prev=""
    local i
    for ((i=0; i<6; i++)); do
        p="$(dirname "$p")"
        [[ "$p" == "$prev" ]] && break
        prev="$p"
        echo "$p"
        if [[ -d "$p/final" ]]; then
            echo "$p/final"
        fi
    done
}

find_llama_server() {
    local root c
    while read -r root; do
        [[ -n "$root" ]] || continue
        for c in \
            "$root/build_server/bin/llama-server" \
            "$root/build/bin/llama-server" \
            "$root/llama.cpp/build_cuda/bin/llama-server" \
            "$root/llama-prism/build_cuda/bin/llama-server" \
            "$root/build_server/bin/Release/llama-server" \
            "$root/build/bin/Release/llama-server"; do
            if [[ -x "$c" ]] || [[ -x "$c.exe" ]]; then
                [[ -x "$c" ]] && echo "$c" || echo "$c.exe"
                return 0
            fi
        done
    done < <({ echo "$ROOT"; sibling_roots; })
    return 1
}

find_model() {
    local root alt
    [[ "${MODEL_PATH:0:1}" == "/" || "${MODEL_PATH:0:2}" == "./" || "${MODEL_PATH:0:2}" == "~/" ]] && {
        [[ -f "$MODEL_PATH" ]] && { echo "$MODEL_PATH"; return 0; }
        return 1
    }
    while read -r root; do
        [[ -n "$root" ]] || continue
        alt="$root/models/$MODEL_PATH"
        if [[ -f "$alt" ]]; then
            echo "$alt"
            return 0
        fi
    done < <({ echo "$ROOT"; sibling_roots; })
    return 1
}

[[ -n "$LLAMA_SERVER" ]] || LLAMA_SERVER="$(find_llama_server)" || {
    echo "ERROR: llama-server not found. Set LLAMA_SERVER=/path/to/llama-server"; exit 1
}
[[ -x "$LLAMA_SERVER" ]] || [[ -x "$LLAMA_SERVER.exe" ]] || {
    echo "ERROR: not executable: $LLAMA_SERVER"; exit 1
}
[[ -x "$LLAMA_SERVER.exe" ]] && LLAMA_SERVER="$LLAMA_SERVER.exe"
MODEL="$(find_model)" || {
    echo "ERROR: model not found ($MODEL_PATH). Run: bash download-model.sh"; exit 1
}

if curl -sf --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
    echo "ERROR: port $PORT already in use - kill the leftover server first."
    exit 1
fi

TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/verify.XXXXXX")"
SERVER_LOG="$TMP_DIR/server.log"
trap '[[ -n "${SERVER_PID:-}" ]] && kill "$SERVER_PID" 2>/dev/null; rm -rf "$TMP_DIR"' EXIT

"$LLAMA_SERVER" \
    -m "$MODEL" -c "$CTX" -ngl "$NGL" -fa on \
    --port "$PORT" --host 127.0.0.1 --no-ui --parallel 1 \
    >"$SERVER_LOG" 2>&1 &
SERVER_PID=$!

for ((i=0; i<240; i++)); do
    if curl -sf --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo "ERROR: server died during startup. Log tail:"
        tail -n 20 "$SERVER_LOG" | sed 's/^/  /'
        exit 1
    fi
    sleep 0.5
done
curl -sf --max-time 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 || {
    echo "ERROR: server did not become ready. Log tail:"
    tail -n 20 "$SERVER_LOG" | sed 's/^/  /'
    exit 1
}
echo "Server ready: $LLAMA_SERVER"
grep -iE "offloaded .* layers|total VRAM" "$SERVER_LOG" 2>/dev/null | head -n 2 | sed 's/^/  /' || true

PROMPTS_LIST=(
    "Can I make the payment next month instead?"
    "I can pay on the 25th of this month."
    "I already paid it yesterday, why are you calling?"
    "I don't have the money right now, I'm in a bad situation."
    "What date should I pay by?"
)

fail=0
for ((p=0; p<PROMPTS; p++)); do
    user_text="${PROMPTS_LIST[$((p % ${#PROMPTS_LIST[@]}))]}"
    body="$(jq -n \
        --arg u "$user_text" \
        '{messages:[{role:"user",content:$u}], n_predict:64, temperature:0.2, seed:42,
          cache_prompt:true, repeat_penalty:1.2, repeat_last_n:256}')"
    code="$(curl -s -o "$TMP_DIR/r$p.json" -w '%{http_code}' --max-time 120 \
        -H 'Content-Type: application/json' -d "$body" \
        "http://127.0.0.1:$PORT/v1/chat/completions" 2>/dev/null || true)"

    text="$(jq -r '.choices[0].message.content // empty' "$TMP_DIR/r$p.json" 2>/dev/null || true)"
    nchars="${#text}"
    printable="$(printf '%s' "$text" | tr -cd '[[:print:]]' | wc -c)"

    ok=1
    reason=""
    [[ "$code" == "200" ]] || { ok=0; reason="HTTP $code"; }
    [[ -n "$text" ]] || { ok=0; reason="empty reply"; }
    (( nchars >= 20 )) || { ok=0; reason="too short ($nchars chars)"; }
    if (( nchars > 0 && printable * 10 / nchars < 9 )); then
        ok=0; reason=">10% non-printable chars (garbled)"
    fi

    echo "  prompt $((p+1)): HTTP $code, ${nchars} chars -> $([ "$ok" -eq 1 ] && echo PASS || echo "FAIL: $reason")"
    [[ "$ok" -eq 1 ]] || fail=$((fail+1))
    printf '      reply: %.80s%s\n' "${text//$'\n'/ }" "$([ "${#text}" -gt 80 ] && echo '...' || true)"
done

echo
if [[ "$fail" -eq 0 ]]; then
    echo "VERIFY PASS: output is legible/coherent on all $PROMPTS prompts."
    echo "Safe to benchmark this build."
    exit 0
else
    echo "VERIFY FAIL: $fail/$PROMPTS replies were garbage or empty."
    echo "Do NOT record speed numbers for this build. Rebuild clean from origin/master."
    exit 1
fi
