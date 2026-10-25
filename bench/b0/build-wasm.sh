#!/usr/bin/env bash
# Builds OpenBLAS and BLIS to WebAssembly (simd128, no relaxed SIMD, one
# thread) and links the B0 driver against each, so the B0 rows compare
# the three libraries in the same engine on the same machine. Needs
# emcc on PATH (the bench workflow installs the Emscripten SDK) and git.
#
#   bench/b0/build-wasm.sh <out-dir>
#
# Produces <out-dir>/b0_openblas.js|.wasm and <out-dir>/b0_blis.js|.wasm;
# run each with `node <out-dir>/b0_<lib>.js <label>`. The library commits
# are pinned below; move them deliberately and re-run both draws.
set -euo pipefail
OUT="${1:?usage: build-wasm.sh <out-dir>}"
HERE="$(cd "$(dirname "$0")" && pwd)"
OPENBLAS_REPO=https://github.com/OpenMathLib/OpenBLAS.git
OPENBLAS_REV=5bab605e847bc491bc90d39baf130b53deaa9a9f   # develop, 2026-10-25
BLIS_REPO=https://github.com/flame/blis.git
BLIS_REV=a8037cfc4330030c726efaec1ffeb1f93d468aa3       # master, 2026-10-06
JOBS="${JOBS:-$(nproc 2>/dev/null || echo 4)}"
LINK="-O3 -msimd128 -s ALLOW_MEMORY_GROWTH=1 -s INITIAL_MEMORY=512MB -s TOTAL_STACK=16MB"
mkdir -p "$OUT"
WORK="$OUT/src"; mkdir -p "$WORK"

fetch() { # repo rev dir
	if [ ! -d "$3/.git" ]; then git init -q "$3"; git -C "$3" remote add origin "$1"; fi
	git -C "$3" fetch -q --depth 1 origin "$2"
	git -C "$3" checkout -q FETCH_HEAD
}

# OpenBLAS: its own WASM128_GENERIC target (hand-written simd128 gemm
# kernels; relaxed SIMD is opt-in and stays off). The CMake path sets
# the target up but does not pass -msimd128 unless told to.
fetch "$OPENBLAS_REPO" "$OPENBLAS_REV" "$WORK/OpenBLAS"
emcmake cmake -S "$WORK/OpenBLAS" -B "$WORK/OpenBLAS/build-wasm" -DNOFORTRAN=1 -DONLY_CBLAS=1 \
	-DBUILD_WITHOUT_LAPACK=1 -DUSE_THREAD=0 -DBUILD_SHARED_LIBS=OFF -DTARGET=WASM128_GENERIC \
	-DCMAKE_BUILD_TYPE=Release -DCMAKE_C_FLAGS="-msimd128" >/dev/null
cmake --build "$WORK/OpenBLAS/build-wasm" -j"$JOBS" >/dev/null
emcc $LINK "$HERE/b0.c" -I "$WORK/OpenBLAS/build-wasm/generated" -I "$WORK/OpenBLAS/build-wasm" -I "$WORK/OpenBLAS" \
	"$WORK/OpenBLAS/build-wasm/lib/libopenblas.a" -o "$OUT/b0_openblas.js"

# BLIS: the generic (plain C) configuration, autovectorized by clang
# under -msimd128; BLIS has no wasm kernels. Its configure records the
# compiler vendor as "emcc", which its makefiles do not know; emcc is
# clang underneath, so the vendor is set to clang.
fetch "$BLIS_REPO" "$BLIS_REV" "$WORK/blis"
( cd "$WORK/blis" && emconfigure ./configure --disable-shared --enable-static --enable-cblas \
	--disable-threading --disable-system CC=emcc CFLAGS="-O3 -msimd128" generic >/dev/null \
	&& sed -i 's/^CC_VENDOR         := emcc/CC_VENDOR         := clang/' config.mk \
	&& make -j"$JOBS" >/dev/null )
emcc $LINK "$HERE/b0.c" -I "$WORK/blis/include/generic" "$WORK/blis/lib/generic/libblis.a" -o "$OUT/b0_blis.js"

# both modules must carry simd128 instructions (0xfd-prefixed opcodes)
for lib in openblas blis; do
	n=$(node -e "const b=require('fs').readFileSync('$OUT/b0_$lib.wasm');let n=0;for(const x of b)if(x===0xfd)n++;console.log(n)")
	echo "b0_$lib.wasm: $(stat -c %s "$OUT/b0_$lib.wasm") bytes, $n simd-prefix bytes"
	[ "$n" -gt 1000 ] || { echo "b0_$lib.wasm has no simd128 code"; exit 1; }
done
