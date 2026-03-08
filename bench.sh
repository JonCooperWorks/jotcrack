#!/usr/bin/env bash
set -euo pipefail

JOTCRACK="./target/release/jotcrack"
WORDLIST="breach.txt"
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
# "found" tokens use key "password" — correctness check + early-exit timing
# "stress" tokens use a key NOT in the wordlist — full scan for throughput

declare -A TOKENS_FOUND=(
  [hs256]="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJqb3RjcmFjay10ZXN0In0.TsGpdKHbCs9gVE-wpUnhNWGsFfsmvLrryksC1Pko1fM"
  [hs384]="eyJhbGciOiJIUzM4NCIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJqb3RjcmFjay10ZXN0In0.lSMwXXnk2AuLn5jEpUVBUrd9pimvBBL-9QYHk0zwI1wRRJFsLog93RF6ypgek97k"
  [hs512]="eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJqb3RjcmFjay10ZXN0In0.DGzZAW-P6w4w_g0o-03Ag9OvtF-gtJad_SbS2wdgeUwwk6cRQcKpOCtKv0qG9pn6rAI8t8Mgg_2OlbB2UzjWGw"
  [a128kw]="eyJhbGciOiJBMTI4S1ciLCJlbmMiOiJBMTI4R0NNIn0.sIvz1-pLW5oZJCyDy8IgOXC8m95RcbWn.QkJCQkJCQkJCQkJC.AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA.AAAAAAAAAAAAAAAAAAAAAA"
  [a192kw]="eyJhbGciOiJBMTkyS1ciLCJlbmMiOiJBMTI4R0NNIn0.NGefuNRFLtB-D7JPJK9nVSJLo8MtQRXD.QkJCQkJCQkJCQkJC.AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA.AAAAAAAAAAAAAAAAAAAAAA"
  [a256kw]="eyJhbGciOiJBMjU2S1ciLCJlbmMiOiJBMTI4R0NNIn0.PbnHRnxspcNUaKv0ij2EAjCqL2l1Lkii.QkJCQkJCQkJCQkJC.AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA.AAAAAAAAAAAAAAAAAAAAAA"
)

declare -A TOKENS_STRESS=(
  [hs256]="eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJqb3RjcmFjay10ZXN0In0.VN0Ygfa_21x4H0YBSWxIxeJLKRyWGxQHrKBIfNBAfno"
  [hs384]="eyJhbGciOiJIUzM4NCIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJqb3RjcmFjay10ZXN0In0.gzdU_A5X7vD1mrBrgwSaVx95Hqtu_5QmqY63y4_7VsCRbMNlz2zQsMPzk7UDSoQz"
  [hs512]="eyJhbGciOiJIUzUxMiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiJqb3RjcmFjay10ZXN0In0.H98XL4zDhHx7eC9pP3s0Vq1KSG6ZTkLXccqf-qjJkvrlz5X9tcwldf105i3ovo9PA5R_WVPwO_26eWvIKguWqA"
  [a128kw]="eyJhbGciOiJBMTI4S1ciLCJlbmMiOiJBMTI4R0NNIn0.mj4JXxDIstTexWtrHpXIQCoHXmP5t6C2.QkJCQkJCQkJCQkJC.AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA.AAAAAAAAAAAAAAAAAAAAAA"
  [a192kw]="eyJhbGciOiJBMTkyS1ciLCJlbmMiOiJBMTI4R0NNIn0.buQbaGhYYYRn7dN6lzaY8LbmbPRzkucF.QkJCQkJCQkJCQkJC.AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA.AAAAAAAAAAAAAAAAAAAAAA"
  [a256kw]="eyJhbGciOiJBMjU2S1ciLCJlbmMiOiJBMTI4R0NNIn0.OXDha23ziFjG44FRnkMekCunhuwYVb6E.QkJCQkJCQkJCQkJC.AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA.AAAAAAAAAAAAAAAAAAAAAA"
)

declare -A SUBCMDS=(
  [hs256]="hs256wordlist"
  [hs384]="hs384wordlist"
  [hs512]="hs512wordlist"
  [a128kw]="jwe-a128kw"
  [a192kw]="jwe-a192kw"
  [a256kw]="jwe-a256kw"
)

ORDER=(hs256 hs384 hs512 a128kw a192kw a256kw)

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

# ── phase 1: correctness (found tokens) ────────────────────────
echo -e "${BOLD}── Phase 1: Correctness (key = \"password\", expect FOUND) ──${RESET}"
echo ""

for alg in "${ORDER[@]}"; do
  run_bench "$alg" "${SUBCMDS[$alg]}" "${TOKENS_FOUND[$alg]}" "$LOGDIR/${alg}_found.log"
done

echo ""

# ── phase 2: full-scan throughput (stress tokens) ───────────────
echo -e "${BOLD}── Phase 2: Full-scan throughput (key NOT in wordlist) ──${RESET}"
echo ""

for alg in "${ORDER[@]}"; do
  run_bench "$alg" "${SUBCMDS[$alg]}" "${TOKENS_STRESS[$alg]}" "$LOGDIR/${alg}_stress.log"
done

echo ""

# ── summary table ───────────────────────────────────────────────
echo -e "${BOLD}── Throughput Summary (full-scan) ──${RESET}"
echo ""
printf "  %-10s │ %14s │ %14s │ %8s\n" "Algorithm" "End-to-End/s" "GPU-Only/s" "Elapsed"
printf "  ──────────┼────────────────┼────────────────┼─────────\n"

for alg in "${ORDER[@]}"; do
  logfile="$LOGDIR/${alg}_stress.log"
  if [[ -f "$logfile" ]]; then
    elapsed=$(extract_stat "elapsed" "$logfile")
    e2e=$(extract_stat "rate_end_to_end" "$logfile")
    gpu_only=$(extract_stat "rate_gpu_only" "$logfile")
    printf "  %-10s │ %14s │ %14s │ %8s\n" "$alg" "$e2e" "$gpu_only" "$elapsed"
  fi
done

echo ""
echo -e "  Full logs saved to ${CYAN}${LOGDIR}/${RESET}"
echo ""
