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

pub mod dgemm;
pub mod dsymm;
pub mod dsyr2k;
pub mod dsyrk;
pub mod dtrmm;
pub mod dtrsm;
pub mod sgemm;
pub mod ssymm;
pub mod ssyr2k;
pub mod ssyrk;
pub mod strmm;
pub mod strsm;
pub mod zgemm;
pub mod zhemm;
pub mod zher2k;
pub mod zherk;
pub mod zsymm;
pub mod zsyr2k;
pub mod zsyrk;
pub mod ztrmm;
pub mod ztrsm;
pub mod cgemm;
pub mod chemm;
pub mod cher2k;
pub mod cherk;
pub mod csymm;
pub mod csyr2k;
pub mod csyrk;
pub mod ctrmm;
pub mod ctrsm;

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

pub use dgemm::{dgemm, dgemm_work, dgemm_packed_work, dgemm_packed_op, dgemm_packed_op_work, dgemm_copy_op, dgemm_copy_op_work, dgemm_colaxpy, dgemm_tiled, dgemm_col4, dgemm_packed, dgemm_packed_bcast};
pub use dsymm::dsymm;
pub use dsyrk::{dsyrk, dsyrk_work, dsyrk_packed, dsyrk_packed_work};
pub use dsyr2k::{dsyr2k, dsyr2k_work};
pub use dtrmm::{dtrmm, dtrmm_work, dtrmm_copy, dtrmm_copy_work, dtrmm_stream, dtrmm_left_packed, dtrmm_left_packed_work};
pub use dtrsm::{dtrsm, dtrsm_work, dtrsm_copy, dtrsm_copy_work, dtrsm_stream, dtrsm_left_packed, dtrsm_left_packed_work};
pub use sgemm::{sgemm, sgemm_work, sgemm_packed_work, sgemm_packed_op, sgemm_packed_op_work, sgemm_copy_op, sgemm_copy_op_work, sgemm_colaxpy, sgemm_tiled, sgemm_col4, sgemm_packed, sgemm_packed_bcast};
pub use ssymm::ssymm;
pub use ssyrk::{ssyrk, ssyrk_work, ssyrk_packed, ssyrk_packed_work};
pub use ssyr2k::{ssyr2k, ssyr2k_work};
pub use strmm::{strmm, strmm_work, strmm_copy, strmm_copy_work, strmm_stream};
pub use strsm::{strsm, strsm_work, strsm_copy, strsm_copy_work, strsm_stream};
pub use zgemm::{zgemm, zgemm_work, zgemm_packed_work, zgemm_packed_op, zgemm_packed_op_work, zgemm_copy_op, zgemm_copy_op_work, zgemm_colaxpy, zgemm_col4, zgemm_packed};
pub use zhemm::zhemm;
pub use zherk::{zherk, zherk_work};
pub use zher2k::{zher2k, zher2k_work};
pub use zsymm::zsymm;
pub use zsyrk::{zsyrk, zsyrk_work};
pub use zsyr2k::{zsyr2k, zsyr2k_work};
pub use ztrmm::{ztrmm, ztrmm_work, ztrmm_copy, ztrmm_copy_work, ztrmm_stream};
pub use ztrsm::{ztrsm, ztrsm_work, ztrsm_copy, ztrsm_copy_work, ztrsm_stream};
pub use cgemm::{cgemm, cgemm_work, cgemm_packed_work, cgemm_packed_op, cgemm_packed_op_work, cgemm_copy_op, cgemm_copy_op_work, cgemm_colaxpy, cgemm_col4, cgemm_packed};
pub use chemm::chemm;
pub use cherk::{cherk, cherk_work};
pub use cher2k::{cher2k, cher2k_work};
pub use csymm::csymm;
pub use csyrk::{csyrk, csyrk_work};
pub use csyr2k::{csyr2k, csyr2k_work};
pub use ctrmm::{ctrmm, ctrmm_work, ctrmm_copy, ctrmm_copy_work, ctrmm_stream};
pub use ctrsm::{ctrsm, ctrsm_work, ctrsm_copy, ctrsm_copy_work, ctrsm_stream};


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
