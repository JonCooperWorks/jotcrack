#!/usr/bin/env bash
set -euo pipefail

MARKOV_ONLY=false
STRESS_ONLY=false
HS_ONLY=false
for arg in "$@"; do
  case "$arg" in
    --markov) MARKOV_ONLY=true ;;
    --stress) STRESS_ONLY=true ;;
    --hs-only) HS_ONLY=true ;;
  esac
done

JOTCRACK="./target/release/jotcrack"
WORDLIST="${WORDLIST:-list.txt}"
LOGDIR="bench_results"
mkdir -p "$LOGDIR"

# ── colour helpers ──────────────────────────────────────────────
GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
BOLD='\033[1m'
RESET='\033[0m'

# ── token definitions ───────────────────────────────────────────
# All tokens use key "password" — correctness check + early-exit timing

declare -A TOKENS_FOUND=(
  [hs256]="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJqb3RjcmFjay10ZXN0In0.TsGpdKHbCs9gVE-wpUnhNWGsFfsmvLrryksC1Pko1fM"
  [hs384]="eyJhbGciOiJIUzM4NCIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJqb3RjcmFjay10ZXN0In0.lSMwXXnk2AuLn5jEpUVBUrd9pimvBBL-9QYHk0zwI1wRRJFsLog93RF6ypgek97k"
  [hs512]="eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJqb3RjcmFjay10ZXN0In0.DGzZAW-P6w4w_g0o-03Ag9OvtF-gtJad_SbS2wdgeUwwk6cRQcKpOCtKv0qG9pn6rAI8t8Mgg_2OlbB2UzjWGw"
  [a128kw]="eyJhbGciOiJBMTI4S1ciLCJlbmMiOiJBMTI4R0NNIn0.sIvz1-pLW5oZJCyDy8IgOXC8m95RcbWn.QkJCQkJCQkJCQkJC.AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA.AAAAAAAAAAAAAAAAAAAAAA"
  [a192kw]="eyJhbGciOiJBMTkyS1ciLCJlbmMiOiJBMTI4R0NNIn0.NGefuNRFLtB-D7JPJK9nVSJLo8MtQRXD.QkJCQkJCQkJCQkJC.AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA.AAAAAAAAAAAAAAAAAAAAAA"
  [a256kw]="eyJhbGciOiJBMjU2S1ciLCJlbmMiOiJBMTI4R0NNIn0.PbnHRnxspcNUaKv0ij2EAjCqL2l1Lkii.QkJCQkJCQkJCQkJC.AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA.AAAAAAAAAAAAAAAAAAAAAA"
)

declare -A SUBCMDS=(
  [hs256]="hs256wordlist"
  [hs384]="hs384wordlist"
  [hs512]="hs512wordlist"
  [a128kw]="jwe-a128kw"
  [a192kw]="jwe-a192kw"
  [a256kw]="jwe-a256kw"
)

if [ "$HS_ONLY" = true ]; then
  ORDER=(hs256 hs384 hs512)
else
  ORDER=(hs256 hs384 hs512 a128kw a192kw a256kw)
fi

# ── helpers ─────────────────────────────────────────────────────
extract_stat() {
  local label="$1" file="$2"
  grep "  ${label}:" "$file" | head -1 | sed "s/.*${label}: //"
}

run_bench() {
  local name="$1" subcmd="$2" token="$3" logfile="$4"
  local stdout_file="${logfile}.stdout"

  printf "  %-12s " "$name"

  "$JOTCRACK" "$subcmd" "$token" --wordlist "$WORDLIST" \
    > "$stdout_file" 2> "$logfile" || true

  local result
  result=$(cat "$stdout_file")
  local tested elapsed e2e gpu_only

  tested=$(extract_stat "tested" "$logfile")
  elapsed=$(extract_stat "elapsed" "$logfile")
  e2e=$(extract_stat "rate_end_to_end" "$logfile")
  gpu_only=$(extract_stat "rate_gpu_only" "$logfile")

  if echo "$result" | grep -q "key:"; then
    printf "${GREEN}FOUND${RESET}    "
  elif echo "$result" | grep -q "NOT FOUND"; then
    printf "${YELLOW}NOT FOUND${RESET}"
  else
    printf "${RED}ERROR${RESET}    "
  fi

  printf "  │ %15s tested │ %8s │ e2e: %14s/s │ gpu: %14s/s\n" \
    "$tested" "$elapsed" "$e2e" "$gpu_only"

  rm -f "$stdout_file"
}

run_markov_bench() {
  local name="$1" token="$2" logfile="$3"
  local stdout_file="${logfile}.stdout"

  printf "  %-12s " "$name"

  "$JOTCRACK" markov "$token" --wordlist "$WORDLIST" \
    --threshold 30 --min-len 8 --max-len 8 \
    > "$stdout_file" 2> "$logfile" || true

  local result
  result=$(cat "$stdout_file")
  local tested elapsed e2e gpu_only

  tested=$(extract_stat "tested" "$logfile")
  elapsed=$(extract_stat "elapsed" "$logfile")
  e2e=$(extract_stat "rate_end_to_end" "$logfile")
  gpu_only=$(extract_stat "rate_gpu_only" "$logfile")

  if echo "$result" | grep -q "key:"; then
    printf "${GREEN}FOUND${RESET}    "
  elif echo "$result" | grep -q "NOT FOUND"; then
    printf "${YELLOW}NOT FOUND${RESET}"
  else
    printf "${RED}ERROR${RESET}    "
  fi

  printf "  │ %15s tested │ %8s │ e2e: %14s/s │ gpu: %14s/s\n" \
    "$tested" "$elapsed" "$e2e" "$gpu_only"

  rm -f "$stdout_file"
}

# ── banner ──────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}╔══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "${BOLD}║              jotcrack benchmark suite                       ║${RESET}"
echo -e "${BOLD}╚══════════════════════════════════════════════════════════════╝${RESET}"
echo ""
echo -e "  binary:   ${CYAN}${JOTCRACK}${RESET}"
echo -e "  wordlist: ${CYAN}${WORDLIST}${RESET}"
echo -e "  date:     $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

if [ "$MARKOV_ONLY" = false ]; then
# ── phase 1: wordlist (key = "password", expect FOUND) ──────────
echo -e "${BOLD}── Phase 1: Wordlist (key = \"password\", expect FOUND) ──${RESET}"
echo ""

for alg in "${ORDER[@]}"; do
  run_bench "$alg" "${SUBCMDS[$alg]}" "${TOKENS_FOUND[$alg]}" "$LOGDIR/${alg}_found.log"
done

echo ""
fi

if [ "$STRESS_ONLY" = false ]; then
# ── phase 2: markov (key = "password", T=30, len=8, expect FOUND) ─
echo -e "${BOLD}── Phase 2: Markov (key = \"password\", T=30, len=8, expect FOUND) ──${RESET}"
echo ""

for alg in "${ORDER[@]}"; do
  run_markov_bench "$alg" "${TOKENS_FOUND[$alg]}" "$LOGDIR/${alg}_markov.log"
done

echo ""
fi

# ── summary table ───────────────────────────────────────────────
echo -e "${BOLD}── Throughput Summary ──${RESET}"
echo ""

if [ "$MARKOV_ONLY" = false ] && [ "$STRESS_ONLY" = false ]; then
  printf "  %-10s │ %14s │ %14s │ %14s │ %14s\n" \
    "Algorithm" "Wordlist E2E/s" "Wordlist GPU/s" "Markov E2E/s" "Markov GPU/s"
  printf "  ──────────┼────────────────┼────────────────┼────────────────┼────────────────\n"

  for alg in "${ORDER[@]}"; do
    found_log="$LOGDIR/${alg}_found.log"
    markov_log="$LOGDIR/${alg}_markov.log"
    wl_e2e=$(extract_stat "rate_end_to_end" "$found_log" 2>/dev/null || echo "N/A")
    wl_gpu=$(extract_stat "rate_gpu_only" "$found_log" 2>/dev/null || echo "N/A")
    mk_e2e=$(extract_stat "rate_end_to_end" "$markov_log" 2>/dev/null || echo "N/A")
    mk_gpu=$(extract_stat "rate_gpu_only" "$markov_log" 2>/dev/null || echo "N/A")
    printf "  %-10s │ %14s │ %14s │ %14s │ %14s\n" \
      "$alg" "$wl_e2e" "$wl_gpu" "$mk_e2e" "$mk_gpu"
  done
elif [ "$STRESS_ONLY" = true ]; then
  printf "  %-10s │ %14s │ %14s\n" \
    "Algorithm" "Wordlist E2E/s" "Wordlist GPU/s"
  printf "  ──────────┼────────────────┼────────────────\n"

  for alg in "${ORDER[@]}"; do
    found_log="$LOGDIR/${alg}_found.log"
    wl_e2e=$(extract_stat "rate_end_to_end" "$found_log" 2>/dev/null || echo "N/A")
    wl_gpu=$(extract_stat "rate_gpu_only" "$found_log" 2>/dev/null || echo "N/A")
    printf "  %-10s │ %14s │ %14s\n" \
      "$alg" "$wl_e2e" "$wl_gpu"
  done
else
  printf "  %-10s │ %14s │ %14s\n" \
    "Algorithm" "Markov E2E/s" "Markov GPU/s"
  printf "  ──────────┼────────────────┼────────────────\n"

  for alg in "${ORDER[@]}"; do
    markov_log="$LOGDIR/${alg}_markov.log"
    mk_e2e=$(extract_stat "rate_end_to_end" "$markov_log" 2>/dev/null || echo "N/A")
    mk_gpu=$(extract_stat "rate_gpu_only" "$markov_log" 2>/dev/null || echo "N/A")
    printf "  %-10s │ %14s │ %14s\n" \
      "$alg" "$mk_e2e" "$mk_gpu"
  done
fi

echo ""
echo -e "  Full logs saved to ${CYAN}${LOGDIR}/${RESET}"
echo ""
