//! Bit-for-bit cross-check of the vendored math (src/math.rs) against
//! libm, the reference it was vendored from (now a dev-dependency, so
//! the shipped crate has no dependencies while this guard runs on
//! every CI push). NaN-producing inputs compare NaN-ness rather than
//! payload bits — IEEE does not pin NaN payloads and hardware sqrt may
//! legitimately differ from software sqrt there; everywhere else the
//! comparison is exact bits.

use blas_wasm::math;

struct Lcg(u64);
impl Lcg {
	fn next(&mut self) -> u64 {
		self.0 = self
			.0
			.wrapping_mul(6364136223846793005)
			.wrapping_add(1442695040888963407);
		self.0
	}
}

fn check_f64(ours: f64, reference: f64, what: &str) {
	if reference.is_nan() {
		assert!(ours.is_nan(), "{what}: expected NaN");
	} else {
		assert_eq!(ours.to_bits(), reference.to_bits(), "{what}");
	}
}

fn check_f32(ours: f32, reference: f32, what: &str) {
	if reference.is_nan() {
		assert!(ours.is_nan(), "{what}: expected NaN");
	} else {
		assert_eq!(ours.to_bits(), reference.to_bits(), "{what}");
	}
}

const F64_EDGES: &[f64] = &[
	0.0,
	-0.0,
	1.0,
	-1.0,
	2.0,
	0.5,
	f64::INFINITY,
	f64::NEG_INFINITY,
	f64::NAN,
	f64::MAX,
	f64::MIN_POSITIVE,
	5e-324,                  // smallest subnormal
	2.2e-308,                // subnormal range
	1e300,
	1e-300,
	1.7976931348623155e308,  // just below MAX
];

const F32_EDGES: &[f32] = &[
	0.0,
	-0.0,
	1.0,
	-1.0,
	2.0,
	0.5,
	f32::INFINITY,
	f32::NEG_INFINITY,
	f32::NAN,
	f32::MAX,
	f32::MIN_POSITIVE,
	1e-45,   // smallest subnormal
	1e38,
	1e-38,
];

#[test]
fn sqrt_bits_match_libm() {
	for &x in F64_EDGES {
		check_f64(math::sqrt(x), libm::sqrt(x), &format!("sqrt({x:e})"));
		check_f64(math::sqrt(-x), libm::sqrt(-x), &format!("sqrt({:e})", -x));
	}
	let mut rng = Lcg(0x5DEECE66D);
	for i in 0..200_000 {
		// random bit patterns cover every exponent incl. subnormals
		let x = f64::from_bits(rng.next()).abs();
		check_f64(math::sqrt(x), libm::sqrt(x), &format!("sqrt sweep #{i} ({x:e})"));
	}
}

#[test]
fn sqrtf_bits_match_libm() {
	for &x in F32_EDGES {
		check_f32(math::sqrtf(x), libm::sqrtf(x), &format!("sqrtf({x:e})"));
		check_f32(math::sqrtf(-x), libm::sqrtf(-x), &format!("sqrtf({:e})", -x));
	}
	let mut rng = Lcg(0xB5DEECE66D);
	for i in 0..200_000 {
		let x = f32::from_bits(rng.next() as u32).abs();
		check_f32(math::sqrtf(x), libm::sqrtf(x), &format!("sqrtf sweep #{i} ({x:e})"));
	}
}

#[test]
fn hypot_bits_match_libm() {
	for &x in F64_EDGES {
		for &y in F64_EDGES {
			for (a, b) in [(x, y), (-x, y), (x, -y), (-x, -y)] {
				check_f64(
					math::hypot(a, b),
					libm::hypot(a, b),
					&format!("hypot({a:e}, {b:e})"),
				);
			}
		}
	}
	let mut rng = Lcg(0xDEADBEEFCAFE);
	for i in 0..200_000 {
		let x = f64::from_bits(rng.next());
		let y = f64::from_bits(rng.next());
		check_f64(math::hypot(x, y), libm::hypot(x, y), &format!("hypot sweep #{i}"));
	}
}

#[test]
fn hypotf_bits_match_libm() {
	for &x in F32_EDGES {
		for &y in F32_EDGES {
			for (a, b) in [(x, y), (-x, y), (x, -y), (-x, -y)] {
				check_f32(
					math::hypotf(a, b),
					libm::hypotf(a, b),
					&format!("hypotf({a:e}, {b:e})"),
				);
			}
		}
	}
	let mut rng = Lcg(0xFACEFEEDF00D);
	for i in 0..200_000 {
		let x = f32::from_bits(rng.next() as u32);
		let y = f32::from_bits(rng.next() as u32);
		check_f32(math::hypotf(x, y), libm::hypotf(x, y), &format!("hypotf sweep #{i}"));
	}
}
