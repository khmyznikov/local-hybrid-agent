#!/usr/bin/env bash
set -euo pipefail

ROOT="${LOCAL_HYBRID_AGENT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
STATE_HOME="${LOCAL_HYBRID_STATE_HOME:-$HOME/.local/share/local-hybrid-agent}"
OPENCLAUDE_ROOT="${OPENCLAUDE_ROOT:-$STATE_HOME/openclaude}"
OPENCLAUDE_REPO="${OPENCLAUDE_REPO:-https://github.com/Gitlawb/openclaude.git}"
OPENCLAUDE_COMMIT=8db8830666bfa115bf6f9586525fae0e3585bf23
PATCH="$ROOT/openclaude_harness/openclaude-0.29.1-local-first.patch"
RUNTIME_DIR="$STATE_HOME/openclaude-runtime"
NODE_DIR="$RUNTIME_DIR/node"

install_node() {
    local temp_dir checksum_line checksum archive extracted
    temp_dir=$(mktemp -d)
    checksum_line=$(curl -fsSL \
        https://nodejs.org/dist/latest-v22.x/SHASUMS256.txt \
        | grep 'node-v.*-linux-arm64.tar.xz$')
    checksum=${checksum_line%% *}
    archive=${checksum_line##* }
    curl -fsSL "https://nodejs.org/dist/latest-v22.x/$archive" \
        -o "$temp_dir/$archive"
    printf '%s  %s\n' "$checksum" "$temp_dir/$archive" | sha256sum -c -
    rm -rf "$NODE_DIR"
    mkdir -p "$RUNTIME_DIR" "$HOME/.local/bin"
    tar -xJf "$temp_dir/$archive" -C "$temp_dir"
    extracted=$(find "$temp_dir" -maxdepth 1 -type d \
        -name 'node-v*-linux-arm64' -print -quit)
    mv "$extracted" "$NODE_DIR"
    ln -sfn "$NODE_DIR/bin/node" "$HOME/.local/bin/node"
    ln -sfn "$NODE_DIR/bin/npm" "$HOME/.local/bin/npm"
    ln -sfn "$NODE_DIR/bin/npx" "$HOME/.local/bin/npx"
    rm -rf "$temp_dir"
}

export PATH="$HOME/.local/bin:$PATH"

if ! command -v node >/dev/null || [[ $(node --version) != v22.* ]]; then
    install_node
fi
if ! command -v bun >/dev/null || [[ $(bun --version) != 1.3.13 ]]; then
    npm install --global --prefix "$HOME/.local" bun@1.3.13
fi

if [[ ! -d "$OPENCLAUDE_ROOT/.git" ]]; then
    mkdir -p "$(dirname "$OPENCLAUDE_ROOT")"
    git clone "$OPENCLAUDE_REPO" "$OPENCLAUDE_ROOT"
fi

if [[ -n $(git -C "$OPENCLAUDE_ROOT" status --porcelain) ]]; then
    if [[ $(git -C "$OPENCLAUDE_ROOT" rev-parse HEAD) != "$OPENCLAUDE_COMMIT" ]] \
        || ! git -C "$OPENCLAUDE_ROOT" apply --reverse --check "$PATCH" \
            2>/dev/null; then
        echo "OpenClaude checkout has unexpected local changes:" >&2
        echo "$OPENCLAUDE_ROOT" >&2
        echo "Set OPENCLAUDE_ROOT to a clean, dedicated path." >&2
        exit 1
    fi
    echo "OpenClaude compatibility patch is already applied."
else
    git -C "$OPENCLAUDE_ROOT" fetch origin "$OPENCLAUDE_COMMIT"
    git -C "$OPENCLAUDE_ROOT" checkout --detach "$OPENCLAUDE_COMMIT"
    git -C "$OPENCLAUDE_ROOT" apply --check "$PATCH"
    git -C "$OPENCLAUDE_ROOT" apply "$PATCH"
fi

cd "$OPENCLAUDE_ROOT"
bun install --frozen-lockfile
bun run build

printf 'OpenClaude: %s\n' "$(node dist/cli.mjs --version)"
printf 'Commit: %s\n' "$(git rev-parse HEAD)"
printf 'Node: %s\n' "$(node --version)"
printf 'Bun: %s\n' "$(bun --version)"
printf 'Checkout: %s\n' "$OPENCLAUDE_ROOT"
