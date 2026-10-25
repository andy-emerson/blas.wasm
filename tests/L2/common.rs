//! Level 2 test helpers: the shared generator and summers plus the
//! matrix shape lists and the exact replays of the dot kernels' folds.

#![allow(dead_code)] // each per-routine test file uses its own subset

#[path = "../shared.rs"]
mod shared;
pub use shared::*;
pub use blas_wasm::{Trans, Uplo};

// (nrows, ncols) shapes: empty, single, tails, unroll boundaries, big
pub const SHAPES: &[(usize, usize)] =
	&[(0, 0), (1, 1), (3, 2), (4, 4), (8, 5), (17, 9), (64, 64), (130, 33)];
// square sizes for the symmetric/triangular ops
pub const NS: &[usize] = &[0, 1, 2, 3, 5, 8, 17, 64, 130];
pub const PADS: &[usize] = &[0, 3];

// ---- exact scalar replays of the Level 1 dot kernels' register folds
// (`src/L1/ddot.rs`, `sdot.rs`, `zdotu.rs`/`zdotc.rs`, `cdotu.rs`/
// `cdotc.rs`): the transposed trmv/trsv kernels reduce with these, so
// their replays must reproduce the fold, not the plain left-to-right
// sum. `dot_folds_match_the_kernels` in `errors.rs` checks each replay
// against the public routine first.

/// `ddot`'s fold: four F64x2 accumulators over chunks of eight, merged
/// (a0 + a1) + (a2 + a3) lane-wise, then lane0 + lane1, then the tail.
pub fn dot_fold_f64(x: &[f64], y: &[f64]) -> f64 {
	let n = x.len();
	let mut acc = [0.0f64; 8];
	let mut i = 0;
	while i + 8 <= n {
		for (k, a) in acc.iter_mut().enumerate() {
			*a += x[i + k] * y[i + k];
		}
		i += 8;
	}
	let l0 = (acc[0] + acc[2]) + (acc[4] + acc[6]);
	let l1 = (acc[1] + acc[3]) + (acc[5] + acc[7]);
	let mut s = l0 + l1;
	while i < n {
		s += x[i] * y[i];
		i += 1;
	}
	s
}

/// `sdot`'s fold: four F32x4 accumulators over chunks of sixteen, merged
/// (a0 + a1) + (a2 + a3) lane-wise, then (l0 + l1) + (l2 + l3), then the tail.
pub fn dot_fold_f32(x: &[f32], y: &[f32]) -> f32 {
	let n = x.len();
	let mut acc = [0.0f32; 16];
	let mut i = 0;
	while i + 16 <= n {
		for (k, a) in acc.iter_mut().enumerate() {
			*a += x[i + k] * y[i + k];
		}
		i += 16;
	}
	let l: Vec<f32> = (0..4).map(|l| (acc[l] + acc[4 + l]) + (acc[8 + l] + acc[12 + l])).collect();
	let mut s = (l[0] + l[1]) + (l[2] + l[3]);
	while i < n {
		s += x[i] * y[i];
		i += 1;
	}
	s
}

/// `zdotu`/`zdotc`'s fold: four complex accumulators over chunks of four,
/// merged (a0 + a1) + (a2 + a3), then the tail; `conj` conjugates `x`.
pub fn dot_fold_c64(x: &[C64], y: &[C64], conj: bool) -> C64 {
	let cj = |v: C64| if conj { v.conj() } else { v };
	let n = x.len();
	let mut acc = [C64::ZERO; 4];
	let mut i = 0;
	while i + 4 <= n {
		for (k, a) in acc.iter_mut().enumerate() {
			*a = *a + cj(x[i + k]) * y[i + k];
		}
		i += 4;
	}
	let mut s = (acc[0] + acc[1]) + (acc[2] + acc[3]);
	while i < n {
		s = s + cj(x[i]) * y[i];
		i += 1;
	}
	s
}

/// `cdotu`/`cdotc`'s fold: four F32x4 accumulators, two complexes each,
/// over chunks of eight, merged (a0 + a1) + (a2 + a3) lane-wise and then
/// the two complex halves added, then the tail; `conj` conjugates `x`.
pub fn dot_fold_c32(x: &[C32], y: &[C32], conj: bool) -> C32 {
	let cj = |v: C32| if conj { v.conj() } else { v };
	let n = x.len();
	let mut acc = [C32::ZERO; 8];
	let mut i = 0;
	while i + 8 <= n {
		for (k, a) in acc.iter_mut().enumerate() {
			*a = *a + cj(x[i + k]) * y[i + k];
		}
		i += 8;
	}
	let half0 = (acc[0] + acc[2]) + (acc[4] + acc[6]);
	let half1 = (acc[1] + acc[3]) + (acc[5] + acc[7]);
	let mut s = half0 + half1;
	while i < n {
		s = s + cj(x[i]) * y[i];
		i += 1;
	}
	s
}
