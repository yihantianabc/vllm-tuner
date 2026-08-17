#!/usr/bin/env bash
set -euo pipefail

MODEL_REPOSITORY="Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION="a09a35458c702b33eeacc393d103063234e8bc28"
MODEL_DIR="${1:-/root/autodl-tmp/models/Qwen2.5-7B-Instruct}"
MODEL_ENDPOINT="${HF_ENDPOINT:-https://huggingface.co}"
BASE_URL="${MODEL_ENDPOINT}/${MODEL_REPOSITORY}/resolve/${MODEL_REVISION}"
CURL_PROXY_ARGS=()
if [[ -n "${ALL_PROXY:-}" ]]; then
  # Remote DNS through socks5h avoids local proxy/DNS certificate mismatches.
  CURL_PROXY_ARGS=(--proxy "${ALL_PROXY}")
fi

FILES=(
  config.json
  generation_config.json
  merges.txt
  model-00001-of-00004.safetensors
  model-00002-of-00004.safetensors
  model-00003-of-00004.safetensors
  model-00004-of-00004.safetensors
  model.safetensors.index.json
  tokenizer.json
  tokenizer_config.json
  vocab.json
)

declare -A EXPECTED_SIZE=(
  [config.json]=663
  [generation_config.json]=243
  [merges.txt]=1671839
  [model-00001-of-00004.safetensors]=3945441440
  [model-00002-of-00004.safetensors]=3864726352
  [model-00003-of-00004.safetensors]=3864726424
  [model-00004-of-00004.safetensors]=3556377672
  [model.safetensors.index.json]=27752
  [tokenizer.json]=7031645
  [tokenizer_config.json]=7305
  [vocab.json]=2776833
)

declare -A EXPECTED_SHA256=(
  [config.json]=7463bb0ea78315365e6c6b74de4e73bbcc8359dfb0c5a737584e077d42c0b03c
  [generation_config.json]=3a8f9087e486054c8a4a08dae2e5a3ba62e23da212b5b8c08bc42cb983c3459f
  [merges.txt]=599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3
  [model-00001-of-00004.safetensors]=a1333e6293854747c481288ea83b348226af178dd565c49b6f9495ba1966aba7
  [model-00002-of-00004.safetensors]=f5d25a2772cb825164a2a2c0fb6d51a87e282abf21e4dd75bc5cfb3cd0ea6185
  [model-00003-of-00004.safetensors]=8efdec4c1bc12317ae1a38dc42b595ce777738a64deea3fcb8a0a91381bcdfd5
  [model-00004-of-00004.safetensors]=1a72d403cdf0c1ec3cb7f289f17b394a01e64394c2e9b3c0f94dbce3faf879bd
  [model.safetensors.index.json]=624bf7c47cd12468fdc16e38a47cf4f19e0415b859a223ba3c027eed2f0e1028
  [tokenizer.json]=c0382117ea329cdf097041132f6d735924b697924d6f6fc3945713e96ce87539
  [tokenizer_config.json]=5b5d4f65d0acd3b2d56a35b56d374a36cbc1c8fa5cf3b3febbbfabf22f359583
  [vocab.json]=ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910
)

mkdir -p "${MODEL_DIR}"

for filename in "${FILES[@]}"; do
  destination="${MODEL_DIR}/${filename}"
  expected_size="${EXPECTED_SIZE[${filename}]}"
  actual_size=0
  if [[ -f "${destination}" ]]; then
    actual_size="$(stat -c '%s' "${destination}")"
  fi
  if [[ "${actual_size}" == "${expected_size}" ]]; then
    actual_sha256="$(sha256sum "${destination}" | awk '{print $1}')"
    if [[ "${actual_sha256}" == "${EXPECTED_SHA256[${filename}]}" ]]; then
      echo "Already complete and verified: ${filename}"
      continue
    fi
    echo "Removing invalid-SHA256 final file: ${filename} (${actual_sha256})"
  elif [[ -f "${destination}" ]]; then
    echo "Removing invalid-size final file: ${filename} (${actual_size}/${expected_size})"
  fi
  if [[ -f "${destination}" ]]; then
    rm -- "${destination}"
  fi
  partial="${destination}.part"
  echo "Downloading ${filename}"
  curl \
    "${CURL_PROXY_ARGS[@]}" \
    --continue-at - \
    --connect-timeout 30 \
    --fail \
    --location \
    --output "${partial}" \
    --retry 20 \
    --retry-all-errors \
    --speed-limit 1024 \
    --speed-time 120 \
    "${BASE_URL}/${filename}?download=true"
  actual_size="$(stat -c '%s' "${partial}")"
  if [[ "${actual_size}" != "${expected_size}" ]]; then
    echo "Size mismatch for ${filename}: expected ${expected_size}, found ${actual_size}" >&2
    exit 1
  fi
  actual_sha256="$(sha256sum "${partial}" | awk '{print $1}')"
  if [[ "${actual_sha256}" != "${EXPECTED_SHA256[${filename}]}" ]]; then
    echo "SHA256 mismatch for ${filename}: ${actual_sha256}" >&2
    exit 1
  fi
  mv -- "${partial}" "${destination}"
done

printf '%s\n' "${MODEL_REVISION}" > "${MODEL_DIR}/.slotune-model-revision"
echo "Verified ${MODEL_REPOSITORY}@${MODEL_REVISION} in ${MODEL_DIR}"
