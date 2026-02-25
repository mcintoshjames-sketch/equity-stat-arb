#!/bin/bash
set -euo pipefail

CONFIG="${STAT_ARB_CONFIG:-/app/config/default.yaml}"

case "${1:-run-live}" in
    run-backtest)
        shift
        exec python -m stat_arb run-backtest --config "$CONFIG" "$@"
        ;;
    run-live)
        shift || true
        BROKER_FLAG=""
        if [ "${BROKER_MODE:-paper}" = "live" ]; then
            BROKER_FLAG="--broker-mode=live"
        fi
        LOOP_FLAG=""
        if [ "${LOOP:-false}" = "true" ]; then
            LOOP_FLAG="--loop"
        fi
        exec python -m stat_arb run-live --config "$CONFIG" $BROKER_FLAG $LOOP_FLAG "$@"
        ;;
    dashboard)
        shift || true
        exec python -m stat_arb dashboard --config "$CONFIG" "$@"
        ;;
    *)
        exec "$@"
        ;;
esac
