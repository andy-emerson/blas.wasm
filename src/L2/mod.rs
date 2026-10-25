//! Level 2: matrix–vector operations, one file per BLAS routine
//! (netlib naming — src/README.md), every one a composition of the
//! Level 1 streams over column slices. The multiply-vector family
//! runs its columns four at a time through the shared blocked kernels
//! (`crate::kernels`); tails and the rank-1/2 updates stay one Level 1
//! call per column. The
//! SIMD hot loops live in `L1/` and `kernels.rs` under their `target_feature`
//! annotations. All four number types follow the same structure
//! (naming: src/README.md).
//!
//! Matrix convention (the whole crate): column-major slice with a
//! column stride — column `j` of an `nrows × ncols` matrix `a` with
//! stride `cs ≥ nrows` is `a[j*cs .. j*cs + nrows]`. Unit row stride
//! only; callers with padded matrices pass the padded column stride as
//! `cs`.




pub mod gemv;
pub mod ger;
pub mod symv;
pub mod syr;
pub mod syr2;
pub mod trmv;
pub mod trsv;

pub use gemv::*;
pub use ger::*;
pub use symv::*;
pub use syr::*;
pub use syr2::*;
pub use trmv::*;
pub use trsv::*;

/// Shared entry checks: the storage really contains an nrows×ncols
/// matrix at stride cs.
#[inline]
pub(crate) fn check_mat(a_len: usize, nrows: usize, ncols: usize, cs: usize) {
	assert!(cs >= nrows, "column stride below row count");
	if nrows > 0 && ncols > 0 {
		assert!(
			a_len >= cs * (ncols - 1) + nrows,
			"matrix storage too short for its dimensions"
		);
	}
}

/// y ← βy with BLAS β=0 semantics (a hard zero-fill, so stale NaN/inf
/// in y cannot leak through 0·y).
#[inline]
pub(crate) fn dscale_y(beta: f64, y: &mut [f64]) {
	if beta == 0.0 {
		y.fill(0.0);
	} else if beta != 1.0 {
		crate::L1::dscal(beta, y);
	}
}

/// f32 twin of `dscale_y`.
#[inline]
pub(crate) fn sscale_y(beta: f32, y: &mut [f32]) {
	if beta == 0.0 {
		y.fill(0.0);
	} else if beta != 1.0 {
		crate::L1::sscal(beta, y);
	}
}

/// c64 twin of `dscale_y` (complex β; β=0 is a hard zero-fill).
#[inline]
pub(crate) fn zscale_y(beta: crate::c64::C64, y: &mut [crate::c64::C64]) {
	use crate::c64::C64;
	if beta == C64::ZERO {
		y.fill(C64::ZERO);
	} else if beta != C64::ONE {
		crate::L1::zscal(beta, y);
	}
}

/// Real-β scale of a complex vector (zherk/zher2k's β is real by
/// definition); β=0 is a hard zero-fill, otherwise one real multiply
/// per component (`zdscal`).
#[inline]
pub(crate) fn zdscale_y(beta: f64, y: &mut [crate::c64::C64]) {
	use crate::c64::C64;
	if beta == 0.0 {
		y.fill(C64::ZERO);
	} else if beta != 1.0 {
		crate::L1::zdscal(beta, y);
	}
}

/// c32 twin of `zscale_y` (complex β; β=0 is a hard zero-fill).
#[inline]
pub(crate) fn cscale_y(beta: crate::c32::C32, y: &mut [crate::c32::C32]) {
	use crate::c32::C32;
	if beta == C32::ZERO {
		y.fill(C32::ZERO);
	} else if beta != C32::ONE {
		crate::L1::cscal(beta, y);
	}
}

/// c32 twin of `zdscale_y` (real β).
#[inline]
pub(crate) fn csscale_y(beta: f32, y: &mut [crate::c32::C32]) {
	use crate::c32::C32;
	if beta == 0.0 {
		y.fill(C32::ZERO);
	} else if beta != 1.0 {
		crate::L1::csscal(beta, y);
	}
}
