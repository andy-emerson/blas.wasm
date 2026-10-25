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
//! Scope today: no-transpose forms only (syrk covers A·Aᵀ). The
//! transposed and conjugate-transposed forms are planned: STATUS.md,
//! todo S1.

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
pub mod ztrmm;
pub mod ztrsm;
pub mod cgemm;
pub mod chemm;
pub mod cher2k;
pub mod cherk;
pub mod ctrmm;
pub mod ctrsm;

pub use dgemm::{dgemm, dgemm_work, dgemm_packed_work, dgemm_colaxpy, dgemm_tiled, dgemm_col4, dgemm_packed};
#[allow(unused_imports)]
pub(crate) use dgemm::dgemm_k;
pub use dsymm::dsymm;
#[allow(unused_imports)]
pub(crate) use dsymm::{dsymm_left_k, dsymm_right_k};
pub use dsyrk::{dsyrk, dsyrk_work};
#[allow(unused_imports)]
pub(crate) use dsyrk::dsyrk_k;
pub use dsyr2k::{dsyr2k, dsyr2k_work};
#[allow(unused_imports)]
pub(crate) use dsyr2k::dsyr2k_k;
pub use dtrmm::{dtrmm, dtrmm_work};
#[allow(unused_imports)]
pub(crate) use dtrmm::{dtrmm_left_k, dtrmm_right_k};
pub use dtrsm::{dtrsm, dtrsm_work};
#[allow(unused_imports)]
pub(crate) use dtrsm::{dtrsm_left_k, dtrsm_right_k};
pub use sgemm::{sgemm, sgemm_work, sgemm_packed_work, sgemm_colaxpy, sgemm_tiled, sgemm_col4, sgemm_packed};
#[allow(unused_imports)]
pub(crate) use sgemm::sgemm_k;
pub use ssymm::ssymm;
#[allow(unused_imports)]
pub(crate) use ssymm::{ssymm_left_k, ssymm_right_k};
pub use ssyrk::{ssyrk, ssyrk_work};
#[allow(unused_imports)]
pub(crate) use ssyrk::ssyrk_k;
pub use ssyr2k::{ssyr2k, ssyr2k_work};
#[allow(unused_imports)]
pub(crate) use ssyr2k::ssyr2k_k;
pub use strmm::{strmm, strmm_work};
#[allow(unused_imports)]
pub(crate) use strmm::{strmm_left_k, strmm_right_k};
pub use strsm::{strsm, strsm_work};
#[allow(unused_imports)]
pub(crate) use strsm::{strsm_left_k, strsm_right_k};
pub use zgemm::{zgemm, zgemm_work, zgemm_packed_work, zgemm_colaxpy, zgemm_packed};
#[allow(unused_imports)]
pub(crate) use zgemm::zgemm_k;
pub use zhemm::zhemm;
#[allow(unused_imports)]
pub(crate) use zhemm::{zhemm_left_k, zhemm_right_k};
pub use zherk::{zherk, zherk_work};
#[allow(unused_imports)]
pub(crate) use zherk::zherk_k;
pub use zher2k::{zher2k, zher2k_work};
#[allow(unused_imports)]
pub(crate) use zher2k::zher2k_k;
pub use ztrmm::{ztrmm, ztrmm_work};
#[allow(unused_imports)]
pub(crate) use ztrmm::{ztrmm_left_k, ztrmm_right_k};
pub use ztrsm::{ztrsm, ztrsm_work};
#[allow(unused_imports)]
pub(crate) use ztrsm::{ztrsm_left_k, ztrsm_right_k};
pub use cgemm::{cgemm, cgemm_work, cgemm_packed_work, cgemm_colaxpy, cgemm_packed};
#[allow(unused_imports)]
pub(crate) use cgemm::cgemm_k;
pub use chemm::chemm;
#[allow(unused_imports)]
pub(crate) use chemm::{chemm_left_k, chemm_right_k};
pub use cherk::{cherk, cherk_work};
#[allow(unused_imports)]
pub(crate) use cherk::cherk_k;
pub use cher2k::{cher2k, cher2k_work};
#[allow(unused_imports)]
pub(crate) use cher2k::cher2k_k;
pub use ctrmm::{ctrmm, ctrmm_work};
#[allow(unused_imports)]
pub(crate) use ctrmm::{ctrmm_left_k, ctrmm_right_k};
pub use ctrsm::{ctrsm, ctrsm_work};
#[allow(unused_imports)]
pub(crate) use ctrsm::{ctrsm_left_k, ctrsm_right_k};


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
