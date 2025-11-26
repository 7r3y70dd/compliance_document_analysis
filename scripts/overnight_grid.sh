#!/usr/bin/env bash
set -euo pipefail

#         /^\/^\
#         _|__|  O|
#\/     /~     \_/ \
# \____|__________/  \
#        \_______      \
#                `\     \                 \
#                  |     |                  \
#                 /      /                    \
#                /     /                       \\
#              /      /                         \ \
#             /     /                            \  \
#           /     /             _----_            \   \
#          /     /           _-~      ~-_         |   |
#         (      (        _-~    _--_    ~-_     _/   |
#          \      ~-____-~    _-~    ~-_    ~-_-~    /
#            ~-_           _-~          ~-_       _-~
#               ~--______-~                ~-___-~

# ---------- COLORS / GLYPHS ----------
if [[ -t 1 ]] && tput colors &>/dev/null; then
  BOLD="$(tput bold)"; DIM="$(tput dim)"; RESET="$(tput sgr0)"
  RED="$(tput setaf 1)"; GREEN="$(tput setaf 2)"; YELLOW="$(tput setaf 3)"
  BLUE="$(tput setaf 4)"; MAGENTA="$(tput setaf 5)"; CYAN="$(tput setaf 6)"
else
  BOLD=""; DIM=""; RESET=""; RED=""; GREEN=""; YELLOW=""; BLUE=""; MAGENTA=""; CYAN=""
fi

CHECK="${GREEN}✔${RESET}"
CROSS="${RED}✖${RESET}"
WARN="${YELLOW}⚠${RESET}"
INFO="${CYAN}ℹ${RESET}"
ROCKET="🚀"
GEAR="⚙"
PEN="✍"

# rotating frames
FRAMES=( "◐" "◓" "◑" "◒" )
SPIN=( "⠋" "⠙" "⠹" "⠸" "⠼" "⠴" "⠦" "⠧" "⠇" "⠏" )
COLORS=( "$RED" "$YELLOW" "$GREEN" "$CYAN" "$BLUE" "$MAGENTA" )

# ---------- BANNER ----------
banner() {
  echo -e "${MAGENTA}${BOLD}"
  cat <<'ASCII'


    ,---,                                                             ____              ,--,
  .'  .' `\                                                         ,'  , `.,-.----.  ,--.'|     ,--,
,---.'     \    ,---.                                ,---.       ,-+-,.' _ |\    /  \ |  | :   ,--.'|                     ,---,
|   |  .`\  |  '   ,'\                              '   ,'\   ,-+-. ;   , |||   :    |:  : '   |  |,                  ,-+-. /  |
:   : |  '  | /   /   |   ,---.             ,---.  /   /   | ,--.'|'   |  |||   | .\ :|  ' |   `--'_      ,--.--.    ,--.'|'   |   ,---.     ,---.
|   ' '  ;  :.   ; ,. :  /     \           /     \.   ; ,. :|   |  ,', |  |,.   : |: |'  | |   ,' ,'|    /       \  |   |  ,"' |  /     \   /     \
'   | ;  .  |'   | |: : /    / '          /    / ''   | |: :|   | /  | |--' |   |  \ :|  | :   '  | |   .--.  .-. | |   | /  | | /    / '  /    /  |
|   | :  |  ''   | .; :.    ' /          .    ' / '   | .; :|   : |  | ,    |   : .  |'  : |__ |  | :    \__\/: . . |   | |  | |.    ' /  .    ' / |
'   : | /  ; |   :    |'   ; :__         '   ; :__|   :    ||   : |  |/     :     |`-'|  | '.'|'  : |__  ," .--.; | |   | |  |/ '   ; :__ '   ;   /|
|   | '` ,/   \   \  / '   | '.'|        '   | '.'|\   \  / |   | |`-'      :   : :   ;  :    ;|  | '.'|/  /  ,.  | |   | |--'  '   | '.'|'   |  / |
;   :  .'      `----'  |   :    :        |   :    : `----'  |   ;/          |   | :   |  ,   / ;  :    ;  :   .'   \|   |/      |   :    :|   :    |
|   ,.'                 \   \  /      ,--,\   \  /          '---'           `---'.|    ---`-'  |  ,   /|  ,     .-./'---'        \   \  /  \   \  /
'---'                    `----'     ,--.'| `----'                   ,--,      `---`             ---`-'  `--`---'                  `----'    `----'
                   ,---,            |  | :                        ,--.'|
               ,-+-. /  |           :  : '              .--.--.   |  |,      .--.--.
   ,--.--.    ,--.'|'   |  ,--.--.  |  ' |        .--, /  /    '  `--'_     /  /    '
  /       \  |   |  ,"' | /       \ '  | |      /_ ./||  :  /`./  ,' ,'|   |  :  /`./
 .--.  .-. | |   | /  | |.--.  .-. ||  | :   , ' , ' :|  :  ;_    '  | |   |  :  ;_
  \__\/: . . |   | |  | | \__\/: . .'  : |__/___/ \: | \  \    `. |  | :    \  \    `.
  ," .--.; | |   | |  |/  ," .--.; ||  | '.'|.  \  ' |  `----.   \'  : |__   `----.   \
 /  /  ,.  | |   | |--'  /  /  ,.  |;  :    ; \  ;   : /  /`--'  /|  | '.'| /  /`--'  /
;  :   .'   \|   |/     ;  :   .'   \  ,   /   \  \  ;'--'.     / ;  :    ;'--'.     /
|  ,     .-./'---'      |  ,     .-./---`-'     :  \  \ `--'---'  |  ,   /   `--'---'
 `--`---'                `--`---'                \  ' ;            ---`-'
                                                  `--`

ASCII
  echo -e "${RESET}"
}

# ---------- TIMING / UI HELPERS ----------
ts() { date -u +"%Y-%m-%dT%H:%M:%SZ"; }
human() {  # robust under set -u
  local T="${1:-0}"
  local H=$(( T / 3600 ))
  local M=$(( (T % 3600) / 60 ))
  local S=$(( T % 60 ))
  printf "%d:%02d:%02d" "$H" "$M" "$S"
}

bar() { # full-width color bar
  local color="${1:-$BLUE}"
  echo -e "${color}════════════════════════════════════════════════════════════════════════════${RESET}"
}

box() {
  local title="$1"
  bar "$MAGENTA"
  echo -e "  ${BOLD}${title}${RESET}"
  bar "$MAGENTA"
}

# ---------- DEVICE TOGGLER (keeps phases on intended CPU/GPU) ----------
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
set_phase_devices() { # llm_device editor_device
  local llm="${1:-"-1"}" ed="${2:-"-1"}"
  export LLM_DEVICE="$llm" EDITOR_DEVICE="$ed"
  if [[ "$llm" == "-1" && "$ed" == "-1" ]]; then
    export CUDA_VISIBLE_DEVICES=""
  else
    local want=() seen=() out=()
    [[ "$llm" != "-1" ]] && want+=("$llm")
    [[ "$ed"  != "-1" ]] && want+=("$ed")
    for g in "${want[@]}"; do
      [[ " ${seen[*]} " == *" $g "* ]] || { seen+=("$g"); out+=("$g"); }
    done
    export CUDA_VISIBLE_DEVICES="$(IFS=,; echo "${out[*]}")"
  fi
}

# ---------- SPINNERS ----------
run_with_spinner() { # label, cmd...
  local label="$1"; shift
  ( eval "$*" ) &
  local pid=$! start=$SECONDS i=0
  printf "%s %s " "${BLUE}${SPIN[0]}${RESET}" "${label}"
  while kill -0 "$pid" 2>/dev/null; do
    i=$(((i+1)%${#SPIN[@]}))
    printf "\r%s %s ${DIM}[%s]${RESET} " "${CYAN}${SPIN[$i]}${RESET}" "${label}" "$(human $((SECONDS-start)))"
    sleep 0.08
  done
  wait "$pid"; local rc=$?
  if (( rc==0 )); then
    printf "\r${CHECK} %s ${DIM}done in %s${RESET}\n" "${label}" "$(human $((SECONDS-start)))"
  else
    printf "\r${CROSS} %s ${DIM}failed after %s (rc=%s)${RESET}\n" "${label}" "$(human $((SECONDS-start)))" "$rc"
  fi
  return $rc
}

# Single colorful wheel for rewrite (one line)
rewrite_with_wheel() { # url outpath
  local url="$1" out="$2"
  ( curl -fsS -X POST "$url" -o "$out" ) &
  local pid=$! start=$SECONDS fi=0 ci=0
  local title="${PEN} ${BOLD}REWRITE${RESET} → ${CYAN}$(basename "$out")${RESET}"
  echo
  bar "$MAGENTA"
  printf "  %s " "$title"
  while kill -0 "$pid" 2>/dev/null; do
    fi=$(((fi+1)%${#FRAMES[@]}))
    ci=$(((ci+1)%${#COLORS[@]}))
    printf "\r  %s %s  ${DIM}[%s]${RESET}   " "${COLORS[$ci]}${FRAMES[$fi]}${RESET}" "$title" "$(human $((SECONDS-start)))"
    sleep 0.12
  done
  wait "$pid"; local rc=$?
  if (( rc==0 )); then
    printf "\r  ${GREEN}${BOLD}✔${RESET} %s ${DIM}completed in %s${RESET}   \n" "$title" "$(human $((SECONDS-start)))"
  else
    printf "\r  ${RED}${BOLD}✖${RESET} %s ${DIM}failed after %s (rc=%s)${RESET}   \n" "$title" "$(human $((SECONDS-start)))" "$rc"
  fi
  bar "$MAGENTA"; echo
  return $rc
}

# ---------- ENV / SETUP ----------
banner
echo -e "${INFO} ${BOLD}Environment bootstrap${RESET}"
source .venv/bin/activate
python -m pip install -U pip >/dev/null
python -m pip install -r requirements.txt >/dev/null

# ------------------------------ CONFIG ------------------------------
BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PORT="${PORT:-8000}"
HOST="${HOST:-0.0.0.0}"
RESULTS_ROOT="${BASE_DIR}/resources/misc"
JSON_RESULTS_DIR="${BASE_DIR}/resources/json_results"
COMPANY_POLICY_DIR="${BASE_DIR}/resources/company_policy"
COMPLIANCE_DIR="${BASE_DIR}/resources/compliance_documents"
POLICY_IN="${COMPANY_POLICY_DIR}/mid_generated_policy.txt"
COMPLIANCE_IN="${COMPLIANCE_DIR}/best_outline.txt"
HEALTH_URL="http://${HOST}:${PORT}/health"

PY="${PYTHON_BIN:-python}"

# Grid knobs (sane, non-explosive combinations)
EMBEDDINGS=(
  "sentence-transformers/all-MiniLM-L6-v2"
  "sentence-transformers/all-mpnet-base-v2"
)
USE_CROSS_ENCODER_FLAGS=("0" "1")
CROSS_ENCODER_MODEL_DEFAULT="cross-encoder/ms-marco-MiniLM-L-6-v2"
USE_SEMANTIC_NORMALIZER_FLAGS=("0" "1")

# Static knobs
SATISFIED_THRESHOLD="${SATISFIED_THRESHOLD:-0.78}"
PARTIAL_THRESHOLD="${PARTIAL_THRESHOLD:-0.60}"
TOP_K_DEFAULT="${TOP_K_DEFAULT:-3}"

# Devices
ANALYZE_LLM_DEVICE="-1"
ANALYZE_EDITOR_DEVICE="-1"
REWRITE_LLM_DEVICE="-1"
REWRITE_EDITOR_DEVICE="${REWRITE_EDITOR_DEVICE:-0}"

# ------------------------------ HELPERS ------------------------------
APP_PID=""
_runlog=""

wait_for_health() {
  local start=$SECONDS
  printf "${BLUE}${GEAR}${RESET} Waiting for ${BOLD}/health${RESET} at ${CYAN}%s${RESET} " "${HEALTH_URL}"
  for _ in {1..60}; do
    if curl -fsS "${HEALTH_URL}" >/dev/null 2>&1; then
      printf "\r${CHECK} Service healthy ${DIM}(%s)${RESET}\n" "$(human $((SECONDS-start)))"
      return 0
    fi
    local idx=$((SECONDS%${#SPIN[@]}))
    printf "\r${YELLOW}${SPIN[$idx]}${RESET} Waiting for /health ${DIM}(%s)${RESET} " "$(human $((SECONDS-start)))"
    sleep 1
  done
  echo
  echo -e "${CROSS} ${RED}ERROR:${RESET} server did not become healthy after ${BOLD}$(human $((SECONDS-start)))${RESET}"
  [[ -n "${_runlog}" && -f "${_runlog}" ]] && tail -n 200 "${_runlog}" || true
  exit 1
}

start_app() {
  local log="$1"
  echo -e "${BLUE}${GEAR}${RESET} Starting app ${DIM}(log: $log)${RESET}"
  nohup "${PY}" -m uvicorn app:app --host "${HOST}" --port "${PORT}" >"$log" 2>&1 &
  APP_PID=$!
  wait_for_health
}

stop_app() {
  if [[ -n "${APP_PID}" ]] && kill -0 "${APP_PID}" 2>/dev/null; then
    echo -e "${YELLOW}⏹${RESET} Stopping app ${DIM}(pid ${APP_PID})${RESET}"
    kill "${APP_PID}" || true
    sleep 1
    if kill -0 "${APP_PID}" 2>/dev/null; then
      echo -e "${WARN} Force killing app"
      kill -9 "${APP_PID}" || true
    fi
  fi
  APP_PID=""
}

make_set_dir() {
  mkdir -p "${RESULTS_ROOT}"
  local next
  next=$(printf "set%02d" "$(( $(find "${RESULTS_ROOT}" -maxdepth 1 -type d -name 'set*' 2>/dev/null | wc -l) + 1 ))")
  local dir="${RESULTS_ROOT}/${next}"
  mkdir -p "${dir}"
  echo "${dir}"
}

record_debug() { local setdir="$1"; shift; printf "[%s] %s\n" "$(ts)" "$*" >> "${setdir}/gen_debug.txt"; }

# ---------- FUN ASCII INTERSTITIALS ----------
ascii_badge() {
  echo -e "${CYAN}"
  cat <<'TAG'
   _______________________
  /                       \
 |  🔧  CONFIG LOCKED IN  |
  \_______________________/
TAG
  echo -e "${RESET}"
}
ascii_finish() {
  echo -e "${GREEN}"
  cat <<'FIN'
                      /^--^\     /^--^\     /^--^\
                      \____/     \____/     \____/
                     /      \   /      \   /      \
                    |        | |        | |        |
                     \__  __/   \__  __/   \__  __/
|^|^|^|^|^|^|^|^|^|^|^|^\ \^|^|^|^/ /^|^|^|^|^\ \^|^|^|^|^|^|^|^|^|^|^|^|
| | | | | | | | | | | | |\ \| | |/ /| | | | | | \ \ | | | | | | | | | | |
########################/ /######\ \###########/ /#######################
| | | | | | | | | | | | \/| | | | \/| | | | | |\/ | | | | | | | | | | | |
|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|_|
FIN
  echo -e "${RESET}"
}

# ------------------------------ MAIN LOOP ------------------------------
SECONDS_TOTAL=0
overall_start="$(ts)"
trap 'stop_app' EXIT

set_idx=0
for EMB in "${EMBEDDINGS[@]}"; do
  for USE_CE in "${USE_CROSS_ENCODER_FLAGS[@]}"; do
    for USE_SEM in "${USE_SEMANTIC_NORMALIZER_FLAGS[@]}"; do
      set_idx=$((set_idx+1))
      setdir="$(make_set_dir)"
      run_start="$(ts)"

      box "SET ${set_idx}  ${ROCKET}  ${DIM}${setdir}${RESET}"
      echo -e "   ${BOLD}EMBEDDING_MODEL${RESET}=${CYAN}${EMB}${RESET}"
      echo -e "   ${BOLD}USE_CROSS_ENCODER${RESET}=${CYAN}${USE_CE}${RESET}"
      echo -e "   ${BOLD}CROSS_ENCODER_MODEL${RESET}=${CYAN}${CROSS_ENCODER_MODEL_DEFAULT}${RESET}"
      echo -e "   ${BOLD}USE_SEMANTIC_NORMALIZER${RESET}=${CYAN}${USE_SEM}${RESET}"
      ascii_badge

      # Write effective config
      cat > "${setdir}/config.json" <<JSON
{
  "timestamp_utc": "${run_start}",
  "embedding_model": "${EMB}",
  "use_cross_encoder": ${USE_CE},
  "cross_encoder_model": "${CROSS_ENCODER_MODEL_DEFAULT}",
  "use_semantic_normalizer": ${USE_SEM},
  "satisfied_threshold": ${SATISFIED_THRESHOLD},
  "partial_threshold": ${PARTIAL_THRESHOLD},
  "top_k_default": ${TOP_K_DEFAULT},
  "analyze_llm_device": ${ANALYZE_LLM_DEVICE},
  "analyze_editor_device": ${ANALYZE_EDITOR_DEVICE},
  "rewrite_llm_device": ${REWRITE_LLM_DEVICE},
  "rewrite_editor_device": ${REWRITE_EDITOR_DEVICE}
}
JSON

      # ---------------- A) ANALYZE original policy ----------------
      bar "$CYAN"
      echo -e "  ${BOLD}${CYAN}🔎 ANALYZE${RESET}"
      export EMBEDDING_MODEL="${EMB}"
      if [[ "${EMB}" == "intfloat/e5-"* ]]; then export USE_E5_PREFIXES=1; else export USE_E5_PREFIXES=0; fi
      export USE_CROSS_ENCODER="${USE_CE}"
      export CROSS_ENCODER_MODEL="${CROSS_ENCODER_MODEL_DEFAULT}"
      export TOP_K_DEFAULT SATISFIED_THRESHOLD PARTIAL_THRESHOLD
      export USE_SEMANTIC_NORMALIZER="${USE_SEM}"
      set_phase_devices "${ANALYZE_LLM_DEVICE}" "${ANALYZE_EDITOR_DEVICE}"
      echo -e "   Devices: LLM=${LLM_DEVICE}, EDITOR=${EDITOR_DEVICE}, CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-<unset>}'"

      _runlog="${setdir}/uvicorn_analyze.log"
      start_app "${_runlog}"
      record_debug "${setdir}" "ANALYZE: app up"

      run_with_spinner "health snapshot" "curl -fsS '${HEALTH_URL}' | jq . > '${setdir}/health_analyze.json'"

      record_debug "${setdir}" "ANALYZE: starting"
      run_with_spinner "analyze-multipart" \
        "curl --fail-with-body -X POST 'http://${HOST}:${PORT}/analyze-multipart' \
           -F 'policy=@${POLICY_IN};type=text/plain' \
           -F 'compliance=@${COMPLIANCE_IN};type=text/plain' \
           -F 'top_k=${TOP_K_DEFAULT}' \
           -F 'use_rationale=true' | jq . > '${setdir}/analyze_result1.json'"

      record_debug "${setdir}" "ANALYZE: done -> ${setdir}/analyze_result1.json"
      mkdir -p "${JSON_RESULTS_DIR}"
      cp -f "${setdir}/analyze_result1.json" "${JSON_RESULTS_DIR}/my_run.json"
      record_debug "${setdir}" "Copied analyze_result1.json -> resources/json_results/my_run.json"

      stop_app
      record_debug "${setdir}" "ANALYZE: app down"

      # ---------------- B) REWRITE (editor on GPU) ----------------
      bar "$MAGENTA"
      echo -e "  ${BOLD}${MAGENTA}${PEN} REWRITE${RESET}"
      set_phase_devices "${REWRITE_LLM_DEVICE}" "${REWRITE_EDITOR_DEVICE}"
      echo -e "   Devices: LLM=${LLM_DEVICE}, EDITOR=${EDITOR_DEVICE}, CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-<unset>}'"

      _runlog="${setdir}/uvicorn_rewrite.log"
      start_app "${_runlog}"
      record_debug "${setdir}" "REWRITE: app up"
      run_with_spinner "health snapshot" "curl -fsS '${HEALTH_URL}' | jq . > '${setdir}/health_rewrite.json'"

      record_debug "${setdir}" "REWRITE: starting"
      REWRITE_URL="http://${HOST}:${PORT}/rewrite-txt-from-json?result=my_run.json&policy=$(basename "${POLICY_IN}")"
      REWRITE_OUT="${setdir}/rewrite01.txt"
      rewrite_with_wheel "${REWRITE_URL}" "${REWRITE_OUT}"
      record_debug "${setdir}" "REWRITE: done -> ${REWRITE_OUT}"

      stop_app
      record_debug "${setdir}" "REWRITE: app down"

      # ---------------- C) RE-ANALYZE rewritten policy -------------
      bar "$GREEN"
      echo -e "  ${BOLD}${GREEN}🔁 RE-ANALYZE${RESET}"
      set_phase_devices "${ANALYZE_LLM_DEVICE}" "${ANALYZE_EDITOR_DEVICE}"
      echo -e "   Devices: LLM=${LLM_DEVICE}, EDITOR=${EDITOR_DEVICE}, CUDA_VISIBLE_DEVICES='${CUDA_VISIBLE_DEVICES:-<unset>}'"

      _runlog="${setdir}/uvicorn_reanalyze.log"
      start_app "${_runlog}"
      record_debug "${setdir}" "RE-ANALYZE: app up"
      run_with_spinner "health snapshot" "curl -fsS '${HEALTH_URL}' | jq . > '${setdir}/health_reanalyze.json'"

      record_debug "${setdir}" "RE-ANALYZE: starting"
      run_with_spinner "analyze-multipart (re)" \
        "curl --fail-with-body -X POST 'http://${HOST}:${PORT}/analyze-multipart' \
           -F 'policy=@${setdir}/rewrite01.txt;type=text/plain' \
           -F 'compliance=@${COMPLIANCE_IN};type=text/plain' \
           -F 'top_k=${TOP_K_DEFAULT}' \
           -F 'use_rationale=true' | jq . > '${setdir}/analyze_result2.json'"

      record_debug "${setdir}" "RE-ANALYZE: done -> ${setdir}/analyze_result2.json"

      stop_app
      record_debug "${setdir}" "RE-ANALYZE: app down"

      # End-of-set summary
      run_end="$(ts)"; run_secs=$SECONDS
      cat >> "${setdir}/gen_debug.txt" <<TXT
----------------------------------------------------------------
SET SUMMARY
Started: ${run_start}
Ended:   ${run_end}
Elapsed: ${run_secs}s

Artifacts:
- ${setdir}/health_analyze.json
- ${setdir}/analyze_result1.json
- ${JSON_RESULTS_DIR}/my_run.json
- ${setdir}/health_rewrite.json
- ${setdir}/rewrite01.txt
- ${setdir}/health_reanalyze.json
- ${setdir}/analyze_result2.json
- ${setdir}/uvicorn_analyze.log
- ${setdir}/uvicorn_rewrite.log
- ${setdir}/uvicorn_reanalyze.log
- ${setdir}/config.json
----------------------------------------------------------------
TXT

      echo -e "${CHECK} Set ${BOLD}${set_idx}${RESET} finished in ${DIM}$(human ${run_secs})${RESET}\n"
      SECONDS=0
    done
  done
done

overall_end="$(ts)"
ascii_finish
echo -e "  ${GREEN}${BOLD}Grid finished${RESET}. Started ${BOLD}${overall_start}${RESET}, ended ${BOLD}${overall_end}${RESET}."
