#!/usr/bin/env bash
# Re-download (pinned revisions) and rebuild data/processed. The processed files are already included in this
# folder, so this is only needed to regenerate them. Requires internet access to huggingface.co.
#   bash scripts/download_data.sh
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TOKENIZER="${TOKENIZER:-deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B}"
python "$REPO_ROOT/scripts/prepare_data.py" --tokenizer "$TOKENIZER" --max_prompt_tokens "${MAX_PROMPT_LENGTH:-1024}" "$@"
