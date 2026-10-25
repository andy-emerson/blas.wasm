//! # axpy — scaled vector addition
//!
//! $$ y \leftarrow \alpha x + y $$
//!
//! for vectors $x, y$ of length $n$ and a scalar $\alpha$. Reference
//! BLAS names the four type variants `daxpy`, `saxpy`, `zaxpy` and
//! `caxpy`. This is `daxpy`, the f64 form; the other three are the
//! same shape at their own lane width, with the complex pair's product
//! form documented in their own modules.
//!
//! ## What each element is
//!
//! Element $i$ of the result is, exactly,
//!
//! $$ y_i \leftarrow \mathrm{fl}\big(y_i + \mathrm{fl}(\alpha x_i)\big), $$
//!
//! one multiply rounding and one add rounding, in that order. There is
//! no fused multiply-add: a fused form rounds once, and relaxed-SIMD
//! fused multiply-add rounds differently on different hardware, which
//! would forfeit the cross-target bit-identity promise. That trade-off
//! is recorded as a deliberately unused lever in `DESIGN.md`.
//!
//! ## Why the SIMD loop is bit-identical to the definition
//!
//! The operation is elementwise: no element's result depends on any
//! other element. A SIMD lane performs the same two roundings on its
//! own element that the scalar definition does, and lanes never
//! combine. So the result is independent of the lane width, of how many
//! elements an iteration processes, and of whether the loop runs on a
//! wasm engine with simd128 or on native hardware through the lane
//! emulation in `lanes.rs`, which performs the lane operations
//! elementwise. The tail loop, which handles the last $n \bmod 4$
//! elements one at a time, performs the same two roundings. Every
//! target therefore produces the same bits, and the test for this
//! routine checks exactly that: bit-for-bit equality against the scalar
//! definition.
//!
//! ## The loop
//!
//! Two f64 lanes per vector register, unrolled twice, so one iteration
//! moves four elements; the `target_feature` attribute on `imp` is what
//! makes the intrinsics compile inline on wasm (without it they compile
//! as out-of-line calls, a measured 6.4× slowdown). The source is
//! below.
//!
//! ## Calling it
//!
//! This example is compiled and run by `cargo test`, so it cannot
//! drift from the code. The only precondition is that the two slices
//! have the same length; a mismatch comes back as an [`Error`] naming
//! argument 3, and nothing is written.
//!
//! ```rust
//! use blas_wasm::L1::daxpy;
//!
//! let x = [1.0, 2.0, 3.0, 4.0, 5.0];
//! let mut y = [10.0, 20.0, 30.0, 40.0, 50.0];
//! daxpy(2.0, &x, &mut y).unwrap();
//! assert_eq!(y, [12.0, 24.0, 36.0, 48.0, 60.0]);
//!
//! let err = daxpy(2.0, &x, &mut y[..3]).unwrap_err();
//! assert_eq!((err.routine, err.arg), ("daxpy", 3));
//! ```
//!
//! ## Measured
//!
//! axpy is a streaming operation and is scored against the fastest
//! memory stream measured in the same run. The current scoreboard, with
//! the runs it came from, is `bench/README.md` in the repository; the
//! per-routine test standard is `tests/README.md`.

use crate::lanes::F64x2;
use crate::Error;

/// `y ← αx + y` over two slices of equal length.
///
/// Errors: `x` and `y` differ in length (argument 3).
pub fn daxpy(alpha: f64, x: &[f64], y: &mut [f64]) -> Result<(), Error> {
	crate::error::same_len("daxpy", 3, x.len(), y.len())?;
	daxpy_k(alpha, x, y);
	Ok(())
}

/// y ← αx + y. Panics on length mismatch.
pub(crate) fn daxpy_k(alpha: f64, x: &[f64], y: &mut [f64]) {
	assert_eq!(x.len(), y.len(), "daxpy: length mismatch");
	unsafe { imp(alpha, x.as_ptr(), y.as_mut_ptr(), x.len()) }
}

#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn imp(alpha: f64, xp: *const f64, yp: *mut f64, len: usize) {
	let va = F64x2::splat(alpha);
	let mut i = 0usize;
	while i + 4 <= len {
		let x0 = F64x2::load(xp.add(i));
		let y0 = F64x2::load(yp.add(i));
		let x1 = F64x2::load(xp.add(i + 2));
		let y1 = F64x2::load(yp.add(i + 2));
		y0.add(x0.mul(va)).store(yp.add(i));
		y1.add(x1.mul(va)).store(yp.add(i + 2));
		i += 4;
	}
	while i < len {
		*yp.add(i) += *xp.add(i) * alpha;
		i += 1;
	}
}
