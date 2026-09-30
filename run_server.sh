#!/usr/bin/env bash
set -e

# Default configuration
QUANT="${QUANT:-native-int4}"
COMPILE=false
SERVER_NAME="127.0.0.1"
SERVER_PORT=8080
LISTEN=""
API_KEY=""
WORKERS=1
MAX_SEQ_LEN="${FISH_MAX_SEQ_LEN:-4096}"
EXTRA_ARGS=()
FIRST_POSITIONAL=true

show_help() {
    echo "============================================================"
    echo " Fish Speech S2 API Server Launcher"
    echo "============================================================"
    echo "Usage: ./run_server.sh [QUANT_VARIANT] [OPTIONS] [-- EXTRA_ARGS...]"
    echo ""
    echo "Quantization Variants:"
    echo "  native-int4                 PyTorch Native Groupwise INT4 g128 (~3.20 GB VRAM) [DEFAULT, FASTEST]"
    echo "  native-int8                 PyTorch Native Weight-Only INT8 (~4.77 GB VRAM)"
    echo "  int4 | torchao-int4         TorchAO Groupwise INT4 g128 (~3.26 GB VRAM)"
    echo "  fp8 | torchao-fp8           TorchAO Dynamic FP8 (~4.77 GB VRAM)"
    echo "  int8 | torchao-int8         TorchAO Dynamic INT8 (~4.77 GB VRAM)"
    echo "  original | bf16 | s2-pro    Original Unquantized BF16 (~9.12 GB VRAM)"
    echo ""
    echo "Options:"
    echo "  -q, --quant <variant>       Specify quantization variant"
    echo "  -c, --compile               Enable torch.compile for faster generation"
    echo "      --no-compile            Disable torch.compile (default)"
    echo "  -l, --max-seq-len <num>     Max KV-cache / context tokens (default: 4096 ≈ 3.2 min)"
    echo "      --context-len <num>     Alias for --max-seq-len"
    echo "      --server-name <ip>      Server host/IP (default: 127.0.0.1)"
    echo "      --host <ip>             Alias for --server-name"
    echo "  -p, --server-port <port>    Server port (default: 8080)"
    echo "      --port <port>           Alias for --server-port"
    echo "      --listen <ip:port>      Direct listen address (e.g. 0.0.0.0:8080)"
    echo "      --api-key <key>         Optional Bearer API key for authorization"
    echo "  -w, --workers <num>         Number of worker processes (default: 1)"
    echo "  -h, --help                  Show this help message"
    echo ""
    echo "Examples:"
    echo "  ./run_server.sh                               # Launch default (native-int4, 4096 ctx)"
    echo "  ./run_server.sh native-int4 -l 2048           # Launch with smaller 2048 ctx (saves VRAM)"
    echo "  ./run_server.sh fp8 --compile                 # Launch compiled FP8"
    echo "  ./run_server.sh fp8 --host 0.0.0.0 -p 8080    # Listen on all network interfaces"
    echo "  ./run_server.sh native-int4 --api-key secret  # Protect server with API key"
    echo "============================================================"
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        -h|--help)
            show_help
            ;;
        -q|--quant|--quantization)
            QUANT="$2"
            FIRST_POSITIONAL=false
            shift 2
            ;;
        -c|--compile)
            COMPILE=true
            shift
            ;;
        --no-compile)
            COMPILE=false
            shift
            ;;
        --server-name|--host)
            SERVER_NAME="$2"
            shift 2
            ;;
        -p|--server-port|--port)
            SERVER_PORT="$2"
            shift 2
            ;;
        -l|--max-seq-len|--context-len)
            MAX_SEQ_LEN="$2"
            shift 2
            ;;
        --listen)
            LISTEN="$2"
            shift 2
            ;;
        --api-key)
            API_KEY="$2"
            shift 2
            ;;
        -w|--workers)
            WORKERS="$2"
            shift 2
            ;;
        --)
            shift
            EXTRA_ARGS+=("$@")
            break
            ;;
        *)
            if [ "$FIRST_POSITIONAL" = true ] && [[ ! "$1" =~ ^- ]]; then
                QUANT="$1"
                FIRST_POSITIONAL=false
            else
                EXTRA_ARGS+=("$1")
            fi
            shift
            ;;
    esac
done

# Resolve checkpoint path
case "$QUANT" in
    fp8|torchao-fp8)
        CHECKPOINT="checkpoints/s2-pro-torchao-fp8"
        ;;
    int8|torchao-int8)
        CHECKPOINT="checkpoints/s2-pro-torchao-int8"
        ;;
    int4|torchao-int4)
        CHECKPOINT="checkpoints/s2-pro-torchao-int4"
        ;;
    native-int8)
        CHECKPOINT="checkpoints/s2-pro-native-int8"
        ;;
    native-int4|native-int4-g128)
        CHECKPOINT="checkpoints/s2-pro-native-int4-g128"
        ;;
    original|bf16|s2-pro|none)
        CHECKPOINT="checkpoints/s2-pro"
        ;;
    *)
        if [ -d "$QUANT" ]; then
            CHECKPOINT="$QUANT"
        else
            echo "Error: Unknown quantization variant '$QUANT'."
            echo "Available variants: fp8, int8, int4, native-int8, native-int4, original"
            echo "Run './run_server.sh --help' to see usage."
            exit 1
        fi
        ;;
esac

if [ ! -d "$CHECKPOINT" ]; then
    echo "Error: Checkpoint directory '$CHECKPOINT' does not exist."
    exit 1
fi

DECODER_PATH="$CHECKPOINT/codec.pth"
if [ ! -f "$DECODER_PATH" ]; then
    DECODER_PATH="checkpoints/s2-pro/codec.pth"
fi

# Determine listen address
if [ -z "$LISTEN" ]; then
    LISTEN="${SERVER_NAME}:${SERVER_PORT}"
fi

# Activate virtual environment
VENV_PATH="$HOME/.venv-fish-audio-s2"
if [ -f "$VENV_PATH/bin/activate" ]; then
    source "$VENV_PATH/bin/activate"
fi

CMD=(python tools/api_server.py \
    --llama-checkpoint-path "$CHECKPOINT" \
    --decoder-checkpoint-path "$DECODER_PATH" \
    --listen "$LISTEN" \
    --workers "$WORKERS")

if [ "$COMPILE" = true ]; then
    CMD+=(--compile)
fi

if [ -n "$API_KEY" ]; then
    CMD+=(--api-key "$API_KEY")
fi

if [ ${#EXTRA_ARGS[@]} -gt 0 ]; then
    CMD+=("${EXTRA_ARGS[@]}")
fi

export FISH_MAX_SEQ_LEN="$MAX_SEQ_LEN"
export FISH_MAX_CODEC_MASK_LEN="$MAX_SEQ_LEN"

echo "============================================================"
echo " Starting Fish Speech API Server"
echo " Checkpoint:  $CHECKPOINT ($QUANT)"
echo " Decoder:     $DECODER_PATH"
echo " Context:     $MAX_SEQ_LEN tokens (~$((MAX_SEQ_LEN / 21))s max single-pass audio)"
echo " Listen:      http://$LISTEN"
echo " API Docs:    http://$LISTEN/docs"
echo " Workers:     $WORKERS"
echo " Compile:     $COMPILE"
[ -n "$API_KEY" ] && echo " Auth:        Bearer token configured"
echo " Command:     ${CMD[*]}"
echo "============================================================"

exec "${CMD[@]}"
