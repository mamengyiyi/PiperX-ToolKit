#!/usr/bin/env bash
# Watchdog for fold_towel dataset rsync jobs (part1 tar.gz + part2/3 zarr).
#
# Usage:
#   ./scripts/watchdog_rsync_datasets.sh                 # run once (cron)
#   ./scripts/watchdog_rsync_datasets.sh --daemon        # background loop
#   ./scripts/watchdog_rsync_datasets.sh --daemon --interval 300
#   ./scripts/watchdog_rsync_datasets.sh --parts 1,2
#
# Cron:
#   */5 * * * * /home/ruihao/PiperX-ToolKit/scripts/watchdog_rsync_datasets.sh >> /home/ruihao/PiperX-ToolKit/datasets/rsync_watchdog.log 2>&1

set -euo pipefail

PROJECT_ROOT="/home/ruihao/PiperX-ToolKit"
DATASETS_DIR="${PROJECT_ROOT}/datasets"
STATE_DIR="${DATASETS_DIR}/.rsync_watchdog_state"
MONITOR_LOG="${DATASETS_DIR}/rsync_watchdog.log"

REMOTE_USER="my"
REMOTE_HOST="ssh-cn-huabei1.ebcloud.com"
REMOTE_PORT="38044"
REMOTE_BASE="/root/data/my/piperx/original_dataset"

SSH_OPTS=(-p "${REMOTE_PORT}" -o ConnectTimeout=30 -o ServerAliveInterval=60 -o ServerAliveCountMax=10 -o BatchMode=yes)
RSYNC_SSH="ssh ${SSH_OPTS[*]}"

MAX_CONCURRENT=3
RESTART_STAGGER_SEC=30
SIZE_COMPLETE_RATIO="0.995"
DAEMON=0
INTERVAL_SEC=300
PART_FILTER=""

log() {
  echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"
}

usage() {
  sed -n '2,11p' "$0" | sed 's/^# \?//'
  exit "${1:-0}"
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --daemon) DAEMON=1; shift ;;
      --interval) INTERVAL_SEC="$2"; shift 2 ;;
      --parts) PART_FILTER="$2"; shift 2 ;;
      -h|--help) usage 0 ;;
      *) log "Unknown arg: $1"; usage 1 ;;
    esac
  done
}

part_enabled() {
  [[ -z "${PART_FILTER}" ]] || [[ ",${PART_FILTER}," == *",$1,"* ]]
}

# part -> kind: tar | zarr
part_kind() {
  case "$1" in
    1) echo "tar" ;;
    2|3) echo "zarr" ;;
    *) return 1 ;;
  esac
}

local_path() {
  case "$1" in
    1) echo "${PROJECT_ROOT}/fold_towel_20260527_my_part1.tar.gz" ;;
    2) echo "${DATASETS_DIR}/fold_towel_20260527_my_part2.zarr" ;;
    3) echo "${DATASETS_DIR}/fold_towel_20260527_my_part3.zarr" ;;
  esac
}

remote_path() {
  case "$1" in
    1) echo "${REMOTE_BASE}/fold_towel_20260527_my_part1.tar.gz" ;;
    2) echo "${REMOTE_BASE}/fold_towel_20260527_my_part2.zarr" ;;
    3) echo "${REMOTE_BASE}/fold_towel_20260527_my_part3.zarr" ;;
  esac
}

log_path() {
  case "$1" in
    1) echo "${DATASETS_DIR}/rsync_part1_tar.log" ;;
    2) echo "${DATASETS_DIR}/rsync_part2_to_new_server.log" ;;
    3) echo "${DATASETS_DIR}/rsync_part3_to_new_server.log" ;;
  esac
}

state_path() {
  echo "${STATE_DIR}/part${1}.state"
}

process_pattern() {
  case "$1" in
    1) echo "rsync.*fold_towel_20260527_my_part1\\.tar\\.gz" ;;
    2) echo "rsync.*fold_towel_20260527_my_part2\\.zarr" ;;
    3) echo "rsync.*fold_towel_20260527_my_part3\\.zarr" ;;
  esac
}

read_state() {
  local f
  f="$(state_path "$1")"
  [[ -f "${f}" ]] && head -n1 "${f}" || echo "UNKNOWN"
}

write_state() {
  mkdir -p "${STATE_DIR}"
  echo "$2" > "$(state_path "$1")"
}

mark_complete() {
  local part="$1" local_b="$2" remote_b="$3"
  write_state "${part}" "COMPLETE"
  {
    echo "verified_at=$(date '+%Y-%m-%d %H:%M:%S')"
    echo "local_bytes=${local_b}"
    echo "remote_bytes=${remote_b}"
  } >> "$(state_path "${part}")"
  log "part${part}: verified complete, stop watching"
}

is_marked_complete() {
  [[ "$(read_state "$1")" == "COMPLETE" ]]
}

any_part_to_watch() {
  for part in 1 2 3; do
    part_enabled "${part}" || continue
    is_marked_complete "${part}" || return 0
  done
  return 1
}

is_running() {
  pgrep -f "$(process_pattern "$1")" >/dev/null 2>&1
}

count_running() {
  local n=0
  for p in 1 2 3; do
    is_marked_complete "${p}" && continue
    is_running "${p}" && n=$((n + 1))
  done
  echo "${n}"
}

local_bytes() {
  if [[ -f "$1" ]]; then
    stat -c '%s' "$1"
  elif [[ -d "$1" ]]; then
    du -sb "$1" 2>/dev/null | awk '{print $1}'
  else
    echo "0"
  fi
}

remote_bytes() {
  ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" \
    "if [[ -f '$1' ]]; then stat -c '%s' '$1'; elif [[ -d '$1' ]]; then du -sb '$1' | awk '{print \$1}'; else echo 0; fi" \
    2>/dev/null || echo "0"
}

size_complete() {
  awk -v l="$1" -v r="$2" -v ratio="${SIZE_COMPLETE_RATIO}" \
    'BEGIN { if (l <= 0) exit 1; exit (r >= l * ratio) ? 0 : 1 }'
}

dry_run_remaining() {
  local local_p="$1" remote_p="$2" kind="$3"
  local src dst output n
  if [[ "${kind}" == "tar" ]]; then
    src="${local_p}"
    dst="${REMOTE_USER}@${REMOTE_HOST}:${remote_p}"
    output="$(rsync -avhn --partial --append-verify -e "${RSYNC_SSH}" "${src}" "${dst}" 2>/dev/null || true)"
    if echo "${output}" | grep -qE '^[^s].+' ; then
      echo "1"
    else
      echo "0"
    fi
    return
  fi

  src="${local_p}/"
  dst="${REMOTE_USER}@${REMOTE_HOST}:${remote_p}/"
  output="$(rsync -avhn --partial --append-verify -e "${RSYNC_SSH}" "${src}" "${dst}" 2>/dev/null || true)"
  n="$(echo "${output}" | awk '/Number of regular files transferred:/ {print $6}' | tr -d ',')"
  if [[ -n "${n}" ]]; then
    echo "${n}"
    return
  fi
  # Completed sync: dry-run only scans metadata, no file lines to transfer.
  if echo "${output}" | grep -q '(DRY RUN)'; then
    echo "0"
  else
    echo "1"
  fi
}

log_tail_error() {
  local log_file
  log_file="$(log_path "$1")"
  [[ -f "${log_file}" ]] || return 1
  if tail -n 8 "${log_file}" | grep -q "rsync error"; then
    log "part${1}: recent rsync error in log"
    tail -n 3 "${log_file}" | while read -r line; do log "  ${line}"; done
    return 0
  fi
  return 1
}

start_rsync() {
  local part="$1" kind local_p log_file running
  kind="$(part_kind "${part}")"
  local_p="$(local_path "${part}")"
  log_file="$(log_path "${part}")"

  if [[ "${kind}" == "tar" && ! -f "${local_p}" ]] || [[ "${kind}" == "zarr" && ! -d "${local_p}" ]]; then
    log "part${part}: local source missing (${local_p})"
    return 1
  fi

  running="$(count_running)"
  if (( running >= MAX_CONCURRENT )); then
    log "part${part}: ${running} jobs already running (max ${MAX_CONCURRENT}), defer"
    return 1
  fi

  log "part${part}: starting rsync (${kind})"
  write_state "${part}" "RUNNING"

  if [[ "${kind}" == "tar" ]]; then
    nohup rsync -avh --progress --partial --append-verify \
      -e "${RSYNC_SSH} -o Compression=no" \
      "${local_p}" \
      "${REMOTE_USER}@${REMOTE_HOST}:$(remote_path "${part}")" \
      >> "${log_file}" 2>&1 &
  else
    nohup rsync -avh --progress --partial --append-verify \
      -e "${RSYNC_SSH}" \
      "${local_p}" \
      "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_BASE}/" \
      >> "${log_file}" 2>&1 &
  fi

  log "part${part}: pid=$! log=${log_file}"
}

check_part() {
  local part="$1" kind local_p remote_p local_b remote_b remaining
  kind="$(part_kind "${part}")"
  local_p="$(local_path "${part}")"
  remote_p="$(remote_path "${part}")"

  # Once verified complete, never check or restart again.
  if is_marked_complete "${part}"; then
    return 0
  fi

  if [[ "${kind}" == "tar" && ! -f "${local_p}" ]] || [[ "${kind}" == "zarr" && ! -d "${local_p}" ]]; then
    log "part${part}: local source missing, skip"
    return 0
  fi

  if is_running "${part}"; then
    log "part${part}: running (${kind}), ok"
    write_state "${part}" "RUNNING"
    return 0
  fi

  local_b="$(local_bytes "${local_p}")"
  remote_b="$(remote_bytes "${remote_p}")"

  log "part${part}: not running (${kind}), local=${local_b} remote=${remote_b}"
  log_tail_error "${part}" || true

  if size_complete "${local_b}" "${remote_b}"; then
    if [[ "${kind}" == "tar" ]]; then
      mark_complete "${part}" "${local_b}" "${remote_b}"
      return 0
    fi
    log "part${part}: sizes match, verifying with dry-run..."
    remaining="$(dry_run_remaining "${local_p}" "${remote_p}" "${kind}")"
    if [[ "${remaining}" == "0" ]]; then
      mark_complete "${part}" "${local_b}" "${remote_b}"
      return 0
    fi
    log "part${part}: dry-run says work remains, restarting"
  else
    log "part${part}: remote incomplete, restarting"
  fi

  start_rsync "${part}"
  sleep "${RESTART_STAGGER_SEC}"
}

run_once() {
  mkdir -p "${STATE_DIR}"

  if ! any_part_to_watch; then
    log "all watched parts complete, nothing to do"
    return 2
  fi

  log "===== watchdog check ====="

  if ! ssh "${SSH_OPTS[@]}" "${REMOTE_USER}@${REMOTE_HOST}" "echo ok" >/dev/null 2>&1; then
    log "ERROR: cannot reach ${REMOTE_HOST}:${REMOTE_PORT}"
    return 1
  fi

  for part in 1 2 3; do
    part_enabled "${part}" || continue
    check_part "${part}"
  done

  log "===== check done ====="

  if ! any_part_to_watch; then
    log "all watched parts complete, watchdog will exit"
    return 2
  fi
  return 0
}

main() {
  parse_args "$@"
  if [[ "${DAEMON}" -eq 1 ]]; then
    log "daemon start interval=${INTERVAL_SEC}s max_concurrent=${MAX_CONCURRENT}"
    while true; do
      run_once
      rc=$?
      if [[ "${rc}" -eq 2 ]]; then
        break
      fi
      sleep "${INTERVAL_SEC}"
    done
    log "daemon stopped"
  else
    run_once || true
  fi
}

main "$@"
