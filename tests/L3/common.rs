//! Level 3 test helpers: the shared generator and summers plus the
//! dimension lists and the explicit `op(X)` copies the flags are
//! defined against.

#![allow(dead_code)] // each per-routine test file uses its own subset

#[path = "../shared.rs"]
mod shared;
pub use shared::*;
pub use blas_wasm::{Diag, Side, Trans, Uplo};

pub const DIMS: &[(usize, usize, usize)] =
	&[(0, 0, 0), (1, 1, 1), (3, 2, 4), (5, 5, 5), (8, 3, 7), (17, 12, 9), (33, 33, 33)];
pub const NS: &[usize] = &[0, 1, 2, 3, 5, 8, 17, 33];
// (m, n) shapes for the in-place triangular replays: tile boundaries
// and tails on both sides
pub const TRI_DIMS: &[(usize, usize)] =
	&[(0, 0), (1, 1), (3, 5), (5, 3), (4, 4), (8, 8), (9, 13), (16, 7), (20, 33), (33, 20)];

/// The explicit copy the flags are defined against: `op(x)` column-major
/// at a leading dimension of rows+1 (deliberately not the tight one);
/// `conj` is applied under `ConjTrans` (the identity for real types).
pub fn op<T: Copy + Default>(x: &[T], ld: usize, rows: usize, cols: usize, t: Trans, conj: impl Fn(T) -> T) -> (Vec<T>, usize) {
	let (r, c) = match t {
		Trans::NoTrans => (rows, cols),
		_ => (cols, rows),
	};
	let ldo = r + 1;
	let mut o = vec![T::default(); ldo * c];
	for j in 0..c {
		for i in 0..r {
			o[j * ldo + i] = match t {
				Trans::NoTrans => x[j * ld + i],
				Trans::Trans => x[i * ld + j],
				Trans::ConjTrans => conj(x[i * ld + j]),
			};
		}
	}
	(o, ldo)
}
pub fn op_f64(x: &[f64], ld: usize, rows: usize, cols: usize, t: Trans) -> (Vec<f64>, usize) {
	op(x, ld, rows, cols, t, |v| v)
}
pub fn op_f32(x: &[f32], ld: usize, rows: usize, cols: usize, t: Trans) -> (Vec<f32>, usize) {
	op(x, ld, rows, cols, t, |v| v)
}
pub fn op_c64(x: &[C64], ld: usize, rows: usize, cols: usize, t: Trans) -> (Vec<C64>, usize) {
	op(x, ld, rows, cols, t, |v| v.conj())
}
pub fn op_c32(x: &[C32], ld: usize, rows: usize, cols: usize, t: Trans) -> (Vec<C32>, usize) {
	op(x, ld, rows, cols, t, |v| v.conj())
}
