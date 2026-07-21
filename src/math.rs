//! Vendored math — the four libm entry points the layer needs (sqrt,
//! sqrtf, hypot, hypotf), brought in-tree so `[dependencies]` is EMPTY
//! (architect decision 2026-07-21: `no_std` stays, consumers inherit
//! zero transitive baggage).
//!
//! sqrt/sqrtf are the hardware instruction via per-architecture
//! intrinsics: IEEE 754 requires sqrt correctly rounded, so every
//! conforming implementation agrees bit-for-bit — swapping libm's out
//! is provably invisible (and the cross-check tests prove it anyway).
//! hypot/hypotf are ported VERBATIM from the libm crate v0.2.16
//! (MIT OR Apache-2.0; itself derived from musl — the same code Rust's
//! own std links on wasm targets), with only the internal sqrt calls
//! redirected to ours.
//!
//! libm remains as a DEV-dependency: `tests/math.rs` cross-checks all
//! four functions against it bit-for-bit on every CI run — LCG sweeps
//! plus the edge cases (±0, ±inf, NaN, subnormals, huge/tiny scaling).
//! Public but `#[doc(hidden)]`: the only intended external caller is
//! that cross-check test.

/// Correctly-rounded f64 square root (the hardware instruction).
#[cfg(target_arch = "wasm32")]
#[inline]
pub fn sqrt(x: f64) -> f64 {
	// lane 0 of the simd128 sqrt — the layer already requires simd128
	// on wasm throughout, so this adds no new requirement
	#[target_feature(enable = "simd128")]
	unsafe fn imp(x: f64) -> f64 {
		use core::arch::wasm32::*;
		f64x2_extract_lane::<0>(f64x2_sqrt(f64x2_splat(x)))
	}
	unsafe { imp(x) }
}

/// Correctly-rounded f32 square root (the hardware instruction).
#[cfg(target_arch = "wasm32")]
#[inline]
pub fn sqrtf(x: f32) -> f32 {
	#[target_feature(enable = "simd128")]
	unsafe fn imp(x: f32) -> f32 {
		use core::arch::wasm32::*;
		f32x4_extract_lane::<0>(f32x4_sqrt(f32x4_splat(x)))
	}
	unsafe { imp(x) }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn sqrt(x: f64) -> f64 {
	// SSE2 is baseline on x86_64
	use core::arch::x86_64::*;
	unsafe { _mm_cvtsd_f64(_mm_sqrt_pd(_mm_set_sd(x))) }
}

#[cfg(target_arch = "x86_64")]
#[inline]
pub fn sqrtf(x: f32) -> f32 {
	use core::arch::x86_64::*;
	unsafe { _mm_cvtss_f32(_mm_sqrt_ss(_mm_set_ss(x))) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn sqrt(x: f64) -> f64 {
	// NEON is baseline on aarch64
	use core::arch::aarch64::*;
	unsafe { vget_lane_f64::<0>(vsqrt_f64(vdup_n_f64(x))) }
}

#[cfg(target_arch = "aarch64")]
#[inline]
pub fn sqrtf(x: f32) -> f32 {
	use core::arch::aarch64::*;
	unsafe { vget_lane_f32::<0>(vsqrt_f32(vdup_n_f32(x))) }
}

#[cfg(not(any(
	target_arch = "wasm32",
	target_arch = "x86_64",
	target_arch = "aarch64"
)))]
compile_error!(
	"blas-wasm's vendored sqrt needs a per-architecture intrinsic; \
	 add one for this target in src/math.rs (any IEEE-conforming sqrt \
	 instruction is bit-identical by definition)"
);

// ---- hypot / hypotf: verbatim libm v0.2.16 ports (logic untouched;
// indentation restyled, `no_panic` attribute dropped, sqrt redirected)

const SPLIT: f64 = 134217728. + 1.; // 0x1p27 + 1 === (2 ^ 27) + 1

// C-style declarations kept verbatim from libm — the port's value is
// being diffable against its source, so no restyling
#[allow(clippy::needless_late_init)]
fn sq(x: f64) -> (f64, f64) {
	let xh: f64;
	let xl: f64;
	let xc: f64;

	xc = x * SPLIT;
	xh = x - xc + xc;
	xl = x - xh;
	let hi = x * x;
	let lo = xh * xh - hi + 2. * xh * xl + xl * xl;
	(hi, lo)
}

/// Guarded f64 Euclidean norm of (x, y) — libm's algorithm.
// C-style declarations kept verbatim from libm — the port's value is
// being diffable against its source, so no restyling
#[allow(clippy::needless_late_init)]
pub fn hypot(mut x: f64, mut y: f64) -> f64 {
	let x1p700 = f64::from_bits(0x6bb0000000000000); // 0x1p700 === 2 ^ 700
	let x1p_700 = f64::from_bits(0x1430000000000000); // 0x1p-700 === 2 ^ -700

	let mut uxi = x.to_bits();
	let mut uyi = y.to_bits();
	let uti;
	let ex: i64;
	let ey: i64;
	let mut z: f64;

	/* arrange |x| >= |y| */
	uxi &= -1i64 as u64 >> 1;
	uyi &= -1i64 as u64 >> 1;
	if uxi < uyi {
		uti = uxi;
		uxi = uyi;
		uyi = uti;
	}

	/* special cases */
	ex = (uxi >> 52) as i64;
	ey = (uyi >> 52) as i64;
	x = f64::from_bits(uxi);
	y = f64::from_bits(uyi);
	/* note: hypot(inf,nan) == inf */
	if ey == 0x7ff {
		return y;
	}
	if ex == 0x7ff || uyi == 0 {
		return x;
	}
	/* note: hypot(x,y) ~= x + y*y/x/2 with inexact for small y/x */
	/* 64 difference is enough for ld80 double_t */
	if ex - ey > 64 {
		return x + y;
	}

	/* precise sqrt argument in nearest rounding mode without overflow */
	/* xh*xh must not overflow and xl*xl must not underflow in sq */
	z = 1.;
	if ex > 0x3ff + 510 {
		z = x1p700;
		x *= x1p_700;
		y *= x1p_700;
	} else if ey < 0x3ff - 450 {
		z = x1p_700;
		x *= x1p700;
		y *= x1p700;
	}
	let (hx, lx) = sq(x);
	let (hy, ly) = sq(y);
	z * sqrt(ly + lx + hy + hx)
}

/// Guarded f32 Euclidean norm of (x, y) — libm's algorithm.
// C-style declarations kept verbatim from libm — the port's value is
// being diffable against its source, so no restyling
#[allow(clippy::needless_late_init)]
pub fn hypotf(mut x: f32, mut y: f32) -> f32 {
	let x1p90 = f32::from_bits(0x6c800000); // 0x1p90f === 2 ^ 90
	let x1p_90 = f32::from_bits(0x12800000); // 0x1p-90f === 2 ^ -90

	let mut uxi = x.to_bits();
	let mut uyi = y.to_bits();
	let uti;
	let mut z: f32;

	uxi &= -1i32 as u32 >> 1;
	uyi &= -1i32 as u32 >> 1;
	if uxi < uyi {
		uti = uxi;
		uxi = uyi;
		uyi = uti;
	}

	x = f32::from_bits(uxi);
	y = f32::from_bits(uyi);
	if uyi == 0xff << 23 {
		return y;
	}
	if uxi >= 0xff << 23 || uyi == 0 || uxi - uyi >= 25 << 23 {
		return x + y;
	}

	z = 1.;
	if uxi >= (0x7f + 60) << 23 {
		z = x1p90;
		x *= x1p_90;
		y *= x1p_90;
	} else if uyi < (0x7f - 60) << 23 {
		z = x1p_90;
		x *= x1p90;
		y *= x1p90;
	}
	z * sqrtf((x as f64 * x as f64 + y as f64 * y as f64) as f32)
}
