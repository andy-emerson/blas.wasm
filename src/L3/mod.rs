//! Level 3: matrix–matrix operations, one file per BLAS routine
//! (netlib naming — src/README.md), every one structurally a loop
//! of Level 1/2 calls over the right-hand matrix's columns. Every op
//! runs its columns four at a time through the shared blocked kernels
//! (`crate::kernels`) or a lockstep column walk — bit-identical to the plain composition
//! wherever the elimination order allows (the two right-side reorders
//! are documented in their files); gemm additionally size-dispatches
//! a register tile. Same matrix convention as Level 2 (column-major
//! slice + column stride, unit row stride).
//!
//! Transposed and conjugate-transposed operands are defined as the
//! no-transpose call on an explicit transpose, and every flagged form
//! is bit-identical to that definition by construction: the syrk
//! family and small gemm transpose the operand into the caller's
//! workspace with `transpose_into` and run the no-transpose shape;
//! large gemm packs from a transposed read (`*gemm_packed_op`); trmm
//! and trsm route per side between a copy and no-copy kernels. Each
//! choice is the verdict of a two-draw runner race (`bench/README.md`,
//! decision D3) and the losing candidate stays exported and tested.


pub mod gemm;
pub mod symm;
pub mod syrk;
pub mod syr2k;
pub mod trmm;
pub mod trsm;

pub use gemm::*;
pub use symm::*;
pub use syrk::*;
pub use syr2k::*;
pub use trmm::*;
pub use trsm::*;

/// `dst ← f(src)ᵀ`: `src` is `rows×cols` at leading dimension `scs`,
/// `dst` receives the `cols×rows` transpose at leading dimension
/// `cols` (so it must hold `rows·cols` elements); `f` is applied to
/// each element (identity, or conjugation for the `ᴴ` forms). Walked in
/// bands of 8 source rows by 256 source columns: the eight destination
/// lines a band writes stay resident across the band, and eight is
/// within the associativity of a Level 1 cache, so a destination stride
/// that is a multiple of 4 KB (every power-of-two dimension) no longer
/// thrashes one cache set. Measured on the container, f64 1024×1024:
/// 4.9 ns per element with 32×32 blocks, 1.5 with these bands; 1000×1000
/// 1.5 and 1.3.
pub(crate) fn transpose_into<T: Copy>(src: &[T], scs: usize, rows: usize, cols: usize, dst: &mut [T], f: impl Fn(T) -> T) {
	const ROWS: usize = 8;
	const COLS: usize = 256;
	assert!(dst.len() >= rows * cols, "transpose_into: destination too short");
	let mut j0 = 0;
	while j0 < cols {
		let j1 = (j0 + COLS).min(cols);
		let mut i0 = 0;
		while i0 < rows {
			let i1 = (i0 + ROWS).min(rows);
			for j in j0..j1 {
				let col = &src[j * scs + i0..j * scs + i1];
				for (i, &v) in (i0..i1).zip(col) {
					dst[i * cols + j] = f(v);
				}
			}
			i0 = i1;
		}
		j0 = j1;
	}
}



pub(crate) use super::L2::check_mat;

/// Symmetric-triangle element lookup: `A[i,j]` with only one triangle
/// stored (f64).
#[inline]
pub(crate) fn dsym_at(a: &[f64], cs: usize, upper: bool, i: usize, j: usize) -> f64 {
	let stored = if upper { i <= j } else { i >= j };
	if stored {
		a[j * cs + i]
	} else {
		a[i * cs + j]
	}
}

/// f32 twin of `dsym_at`.
#[inline]
pub(crate) fn ssym_at(a: &[f32], cs: usize, upper: bool, i: usize, j: usize) -> f32 {
	let stored = if upper { i <= j } else { i >= j };
	if stored {
		a[j * cs + i]
	} else {
		a[i * cs + j]
	}
}

/// The `(i, j)` element of a complex SYMMETRIC matrix with one triangle
/// stored: the stored element, or its mirror unconjugated.
#[inline]
pub(crate) fn zsym_at(a: &[crate::c64::C64], cs: usize, upper: bool, i: usize, j: usize) -> crate::c64::C64 {
	let stored = if upper { i <= j } else { i >= j };
	if stored { a[j * cs + i] } else { a[i * cs + j] }
}

/// The c32 twin of [`zsym_at`].
#[inline]
pub(crate) fn csym_at(a: &[crate::c32::C32], cs: usize, upper: bool, i: usize, j: usize) -> crate::c32::C32 {
	let stored = if upper { i <= j } else { i >= j };
	if stored { a[j * cs + i] } else { a[i * cs + j] }
}

/// Hermitian-triangle element lookup: `A[i,j]` with only one triangle
/// stored — conjugated on the reflected side, and REAL on the
/// diagonal (stored diagonal imaginary parts are ignored, per the
/// LAPACK Hermitian storage convention).
#[inline]
pub(crate) fn zher_at(
	a: &[crate::c64::C64],
	cs: usize,
	upper: bool,
	i: usize,
	j: usize,
) -> crate::c64::C64 {
	use crate::c64::C64;
	if i == j {
		return C64::new(a[j * cs + j].re, 0.0);
	}
	let stored = if upper { i < j } else { i > j };
	if stored {
		a[j * cs + i]
	} else {
		a[i * cs + j].conj()
	}
}

/// c32 twin of `zher_at`.
#[inline]
pub(crate) fn cher_at(
	a: &[crate::c32::C32],
	cs: usize,
	upper: bool,
	i: usize,
	j: usize,
) -> crate::c32::C32 {
	use crate::c32::C32;
	if i == j {
		return C32::new(a[j * cs + j].re, 0.0);
	}
	let stored = if upper { i < j } else { i > j };
	if stored {
		a[j * cs + i]
	} else {
		a[i * cs + j].conj()
	}
}
