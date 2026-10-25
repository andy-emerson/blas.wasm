//! The `blas.wasm` C ABI — every routine of the layer exported as a
//! flat-pointer `extern "C"` function, compiled to a standalone wasm
//! module for JavaScript and any non-Rust consumer. (Rust consumers
//! should depend on the `blas-wasm` crate directly instead: same code,
//! cross-crate inlining, no ABI boundary.)
//!
//! Conventions (mirroring the crate, netlib heritage):
//! - column-major matrices, `cs`/`acs`/`bcs`/`ccs` = column stride;
//!   a rows x cols matrix pointer must be valid for exactly
//!   `cs*(cols-1) + rows` elements; every argument error the crate
//!   reports (`Err`) TRAPS here - panics abort in wasm; the error slot
//!   is STATUS.md todo S6. The checks are netlib's: `cs >= max(1,
//!   rows)`, so an empty matrix still needs `cs >= 1`;
//! - `upper`/`unit` flags are u32, 0 = false, nonzero = true;
//! - complex scalars pass as (re, im) argument pairs; complex arrays
//!   are interleaved re,im pairs (C64 = 2xf64, C32 = 2xf32, the C99 /
//!   Fortran layout); complex-valued results write through an out
//!   pointer;
//! - `wasm_alloc`/`wasm_free` are provided so a host without its own
//!   view of linear memory can create buffers (16-byte aligned).
//!
//! Written mechanically from the crate's public signatures and
//! maintained by hand - one wrapper per
//! routine, strictly uniform shape; keep new wrappers to the same
//! pattern.

use blas_wasm::{C32, C64};

use blas_wasm::{Diag, Side, Trans, Uplo};
fn uplo(upper: bool) -> Uplo {
	if upper { Uplo::Upper } else { Uplo::Lower }
}
fn diag(unit: bool) -> Diag {
	if unit { Diag::Unit } else { Diag::NonUnit }
}
use blas_wasm::{L1, L2, L3};

#[inline]
fn ml(rows: usize, cols: usize, cs: usize) -> usize {
	if cols == 0 { 0 } else { cs * (cols - 1) + rows }
}
/// # Safety
/// `p` must be valid for `len` elements.
unsafe fn s<'a, T>(p: *const T, len: usize) -> &'a [T] {
	core::slice::from_raw_parts(p, len)
}
/// # Safety
/// `p` must be valid for `len` elements, exclusively.
unsafe fn sm<'a, T>(p: *mut T, len: usize) -> &'a mut [T] {
	core::slice::from_raw_parts_mut(p, len)
}

#[no_mangle]
pub extern "C" fn wasm_alloc(bytes: usize) -> *mut u8 {
	let layout = core::alloc::Layout::from_size_align(bytes.max(1), 16).unwrap();
	unsafe { std::alloc::alloc(layout) }
}

/// # Safety
/// `(ptr, bytes)` must come from `wasm_alloc` with the same size.
#[no_mangle]
pub unsafe extern "C" fn wasm_free(ptr: *mut u8, bytes: usize) {
	let layout = core::alloc::Layout::from_size_align(bytes.max(1), 16).unwrap();
	unsafe { std::alloc::dealloc(ptr, layout) }
}

// ---------------- d-routines (f64) ----------------

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn dcopy(x: *const f64, y: *mut f64, n: usize) {
	L1::dcopy(unsafe { s(x, n) }, unsafe { sm(y, n) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn dswap(x: *mut f64, y: *mut f64, n: usize) {
	L1::dswap(unsafe { sm(x, n) }, unsafe { sm(y, n) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn dscal(alpha: f64, x: *mut f64, n: usize) {
	L1::dscal(alpha, unsafe { sm(x, n) });
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn daxpy(alpha: f64, x: *const f64, y: *mut f64, n: usize) {
	L1::daxpy(alpha, unsafe { s(x, n) }, unsafe { sm(y, n) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn drot(x: *mut f64, y: *mut f64, c: f64, s_: f64, n: usize) {
	L1::drot(unsafe { sm(x, n) }, unsafe { sm(y, n) }, c, s_).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn ddot(x: *const f64, y: *const f64, n: usize) -> f64 {
	L1::ddot(unsafe { s(x, n) }, unsafe { s(y, n) }).unwrap()
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn dnrm2(x: *const f64, n: usize) -> f64 {
	L1::dnrm2(unsafe { s(x, n) })
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn dasum(x: *const f64, n: usize) -> f64 {
	L1::dasum(unsafe { s(x, n) })
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn idamax(x: *const f64, n: usize) -> usize {
	L1::idamax(unsafe { s(x, n) })
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn drotg(a: f64, b: f64, c: *mut f64, s_: *mut f64, r: *mut f64) {
	let g = L1::drotg(a, b);
	unsafe {
		*c = g.c;
		*s_ = g.s;
		*r = g.r;
	}
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn dgemv(alpha: f64, nrows: usize, ncols: usize, a: *const f64, cs: usize, x: *const f64, beta: f64, y: *mut f64) {
	L2::dgemv(Trans::NoTrans, nrows, ncols, alpha, unsafe { s(a, ml(nrows, ncols, cs)) }, cs, unsafe { s(x, ncols) }, beta, unsafe { sm(y, nrows) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn dgemv_t(alpha: f64, nrows: usize, ncols: usize, a: *const f64, cs: usize, x: *const f64, beta: f64, y: *mut f64) {
	L2::dgemv(Trans::Trans, nrows, ncols, alpha, unsafe { s(a, ml(nrows, ncols, cs)) }, cs, unsafe { s(x, nrows) }, beta, unsafe { sm(y, ncols) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn dger(alpha: f64, nrows: usize, ncols: usize, a: *mut f64, cs: usize, x: *const f64, y: *const f64) {
	L2::dger(nrows, ncols, alpha, unsafe { s(x, nrows) }, unsafe { s(y, ncols) }, unsafe { sm(a, ml(nrows, ncols, cs)) }, cs).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn dsymv(alpha: f64, n: usize, a: *const f64, cs: usize, upper: u32, x: *const f64, beta: f64, y: *mut f64) {
	L2::dsymv(uplo(upper != 0), n, alpha, unsafe { s(a, ml(n, n, cs)) }, cs, unsafe { s(x, n) }, beta, unsafe { sm(y, n) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn dsyr(alpha: f64, n: usize, a: *mut f64, cs: usize, upper: u32, x: *const f64) {
	L2::dsyr(uplo(upper != 0), n, alpha, unsafe { s(x, n) }, unsafe { sm(a, ml(n, n, cs)) }, cs).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn dsyr2(alpha: f64, n: usize, a: *mut f64, cs: usize, upper: u32, x: *const f64, y: *const f64) {
	L2::dsyr2(uplo(upper != 0), n, alpha, unsafe { s(x, n) }, unsafe { s(y, n) }, unsafe { sm(a, ml(n, n, cs)) }, cs).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn dtrmv(n: usize, a: *const f64, cs: usize, upper: u32, unit: u32, x: *mut f64) {
	L2::dtrmv(uplo(upper != 0), Trans::NoTrans, diag(unit != 0), n, unsafe { s(a, ml(n, n, cs)) }, cs, unsafe { sm(x, n) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn dtrsv(n: usize, a: *const f64, cs: usize, upper: u32, unit: u32, x: *mut f64) {
	L2::dtrsv(uplo(upper != 0), Trans::NoTrans, diag(unit != 0), n, unsafe { s(a, ml(n, n, cs)) }, cs, unsafe { sm(x, n) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn dgemm(alpha: f64, m: usize, k: usize, n: usize, a: *const f64, acs: usize, b: *const f64, bcs: usize, beta: f64, c: *mut f64, ccs: usize) {
	L3::dgemm(Trans::NoTrans, Trans::NoTrans, m, n, k, alpha, unsafe { s(a, ml(m, k, acs)) }, acs, unsafe { s(b, ml(k, n, bcs)) }, bcs, beta, unsafe { sm(c, ml(m, n, ccs)) }, ccs, &mut vec![0.0f64; L3::dgemm_work(Trans::NoTrans, Trans::NoTrans, m, n, k)]).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn dsymm_left(alpha: f64, m: usize, n: usize, a: *const f64, acs: usize, upper: u32, b: *const f64, bcs: usize, beta: f64, c: *mut f64, ccs: usize) {
	L3::dsymm(Side::Left, uplo(upper != 0), m, n, alpha, unsafe { s(a, ml(m, m, acs)) }, acs, unsafe { s(b, ml(m, n, bcs)) }, bcs, beta, unsafe { sm(c, ml(m, n, ccs)) }, ccs).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn dsymm_right(alpha: f64, m: usize, n: usize, a: *const f64, acs: usize, upper: u32, b: *const f64, bcs: usize, beta: f64, c: *mut f64, ccs: usize) {
	L3::dsymm(Side::Right, uplo(upper != 0), m, n, alpha, unsafe { s(a, ml(n, n, acs)) }, acs, unsafe { s(b, ml(m, n, bcs)) }, bcs, beta, unsafe { sm(c, ml(m, n, ccs)) }, ccs).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn dsyrk(alpha: f64, n: usize, k: usize, a: *const f64, acs: usize, beta: f64, c: *mut f64, ccs: usize, upper: u32) {
	L3::dsyrk(uplo(upper != 0), Trans::NoTrans, n, k, alpha, unsafe { s(a, ml(n, k, acs)) }, acs, beta, unsafe { sm(c, ml(n, n, ccs)) }, ccs, &mut vec![0.0f64; L3::dsyrk_work(n, k)]).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn dsyr2k(alpha: f64, n: usize, k: usize, a: *const f64, acs: usize, b: *const f64, bcs: usize, beta: f64, c: *mut f64, ccs: usize, upper: u32) {
	L3::dsyr2k(uplo(upper != 0), Trans::NoTrans, n, k, alpha, unsafe { s(a, ml(n, k, acs)) }, acs, unsafe { s(b, ml(n, k, bcs)) }, bcs, beta, unsafe { sm(c, ml(n, n, ccs)) }, ccs, &mut vec![0.0f64; L3::dsyr2k_work(n, k)]).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn dtrmm_left(alpha: f64, m: usize, n: usize, a: *const f64, acs: usize, upper: u32, unit: u32, b: *mut f64, bcs: usize) {
	L3::dtrmm(Side::Left, uplo(upper != 0), Trans::NoTrans, diag(unit != 0), m, n, alpha, unsafe { s(a, ml(m, m, acs)) }, acs, unsafe { sm(b, ml(m, n, bcs)) }, bcs, &mut vec![0.0f64; L3::dtrmm_work(Side::Left, Trans::NoTrans, m, n)]).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn dtrmm_right(alpha: f64, m: usize, n: usize, a: *const f64, acs: usize, upper: u32, unit: u32, b: *mut f64, bcs: usize) {
	L3::dtrmm(Side::Right, uplo(upper != 0), Trans::NoTrans, diag(unit != 0), m, n, alpha, unsafe { s(a, ml(n, n, acs)) }, acs, unsafe { sm(b, ml(m, n, bcs)) }, bcs, &mut vec![0.0f64; L3::dtrmm_work(Side::Right, Trans::NoTrans, m, n)]).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn dtrsm_left(alpha: f64, m: usize, n: usize, a: *const f64, acs: usize, upper: u32, unit: u32, b: *mut f64, bcs: usize) {
	L3::dtrsm(Side::Left, uplo(upper != 0), Trans::NoTrans, diag(unit != 0), m, n, alpha, unsafe { s(a, ml(m, m, acs)) }, acs, unsafe { sm(b, ml(m, n, bcs)) }, bcs, &mut vec![0.0f64; L3::dtrsm_work(Side::Left, Trans::NoTrans, m, n)]).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn dtrsm_right(alpha: f64, m: usize, n: usize, a: *const f64, acs: usize, upper: u32, unit: u32, b: *mut f64, bcs: usize) {
	L3::dtrsm(Side::Right, uplo(upper != 0), Trans::NoTrans, diag(unit != 0), m, n, alpha, unsafe { s(a, ml(n, n, acs)) }, acs, unsafe { sm(b, ml(m, n, bcs)) }, bcs, &mut vec![0.0f64; L3::dtrsm_work(Side::Right, Trans::NoTrans, m, n)]).unwrap();
}

// ---------------- s-routines (f32) ----------------

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn scopy(x: *const f32, y: *mut f32, n: usize) {
	L1::scopy(unsafe { s(x, n) }, unsafe { sm(y, n) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn sswap(x: *mut f32, y: *mut f32, n: usize) {
	L1::sswap(unsafe { sm(x, n) }, unsafe { sm(y, n) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn sscal(alpha: f32, x: *mut f32, n: usize) {
	L1::sscal(alpha, unsafe { sm(x, n) });
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn saxpy(alpha: f32, x: *const f32, y: *mut f32, n: usize) {
	L1::saxpy(alpha, unsafe { s(x, n) }, unsafe { sm(y, n) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn srot(x: *mut f32, y: *mut f32, c: f32, s_: f32, n: usize) {
	L1::srot(unsafe { sm(x, n) }, unsafe { sm(y, n) }, c, s_).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn sdot(x: *const f32, y: *const f32, n: usize) -> f32 {
	L1::sdot(unsafe { s(x, n) }, unsafe { s(y, n) }).unwrap()
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn snrm2(x: *const f32, n: usize) -> f32 {
	L1::snrm2(unsafe { s(x, n) })
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn sasum(x: *const f32, n: usize) -> f32 {
	L1::sasum(unsafe { s(x, n) })
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn isamax(x: *const f32, n: usize) -> usize {
	L1::isamax(unsafe { s(x, n) })
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn srotg(a: f32, b: f32, c: *mut f32, s_: *mut f32, r: *mut f32) {
	let g = L1::srotg(a, b);
	unsafe {
		*c = g.c;
		*s_ = g.s;
		*r = g.r;
	}
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn sgemv(alpha: f32, nrows: usize, ncols: usize, a: *const f32, cs: usize, x: *const f32, beta: f32, y: *mut f32) {
	L2::sgemv(Trans::NoTrans, nrows, ncols, alpha, unsafe { s(a, ml(nrows, ncols, cs)) }, cs, unsafe { s(x, ncols) }, beta, unsafe { sm(y, nrows) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn sgemv_t(alpha: f32, nrows: usize, ncols: usize, a: *const f32, cs: usize, x: *const f32, beta: f32, y: *mut f32) {
	L2::sgemv(Trans::Trans, nrows, ncols, alpha, unsafe { s(a, ml(nrows, ncols, cs)) }, cs, unsafe { s(x, nrows) }, beta, unsafe { sm(y, ncols) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn sger(alpha: f32, nrows: usize, ncols: usize, a: *mut f32, cs: usize, x: *const f32, y: *const f32) {
	L2::sger(nrows, ncols, alpha, unsafe { s(x, nrows) }, unsafe { s(y, ncols) }, unsafe { sm(a, ml(nrows, ncols, cs)) }, cs).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn ssymv(alpha: f32, n: usize, a: *const f32, cs: usize, upper: u32, x: *const f32, beta: f32, y: *mut f32) {
	L2::ssymv(uplo(upper != 0), n, alpha, unsafe { s(a, ml(n, n, cs)) }, cs, unsafe { s(x, n) }, beta, unsafe { sm(y, n) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn ssyr(alpha: f32, n: usize, a: *mut f32, cs: usize, upper: u32, x: *const f32) {
	L2::ssyr(uplo(upper != 0), n, alpha, unsafe { s(x, n) }, unsafe { sm(a, ml(n, n, cs)) }, cs).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn ssyr2(alpha: f32, n: usize, a: *mut f32, cs: usize, upper: u32, x: *const f32, y: *const f32) {
	L2::ssyr2(uplo(upper != 0), n, alpha, unsafe { s(x, n) }, unsafe { s(y, n) }, unsafe { sm(a, ml(n, n, cs)) }, cs).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn strmv(n: usize, a: *const f32, cs: usize, upper: u32, unit: u32, x: *mut f32) {
	L2::strmv(uplo(upper != 0), Trans::NoTrans, diag(unit != 0), n, unsafe { s(a, ml(n, n, cs)) }, cs, unsafe { sm(x, n) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn strsv(n: usize, a: *const f32, cs: usize, upper: u32, unit: u32, x: *mut f32) {
	L2::strsv(uplo(upper != 0), Trans::NoTrans, diag(unit != 0), n, unsafe { s(a, ml(n, n, cs)) }, cs, unsafe { sm(x, n) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn sgemm(alpha: f32, m: usize, k: usize, n: usize, a: *const f32, acs: usize, b: *const f32, bcs: usize, beta: f32, c: *mut f32, ccs: usize) {
	L3::sgemm(Trans::NoTrans, Trans::NoTrans, m, n, k, alpha, unsafe { s(a, ml(m, k, acs)) }, acs, unsafe { s(b, ml(k, n, bcs)) }, bcs, beta, unsafe { sm(c, ml(m, n, ccs)) }, ccs, &mut vec![0.0f32; L3::sgemm_work(Trans::NoTrans, Trans::NoTrans, m, n, k)]).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn ssymm_left(alpha: f32, m: usize, n: usize, a: *const f32, acs: usize, upper: u32, b: *const f32, bcs: usize, beta: f32, c: *mut f32, ccs: usize) {
	L3::ssymm(Side::Left, uplo(upper != 0), m, n, alpha, unsafe { s(a, ml(m, m, acs)) }, acs, unsafe { s(b, ml(m, n, bcs)) }, bcs, beta, unsafe { sm(c, ml(m, n, ccs)) }, ccs).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn ssymm_right(alpha: f32, m: usize, n: usize, a: *const f32, acs: usize, upper: u32, b: *const f32, bcs: usize, beta: f32, c: *mut f32, ccs: usize) {
	L3::ssymm(Side::Right, uplo(upper != 0), m, n, alpha, unsafe { s(a, ml(n, n, acs)) }, acs, unsafe { s(b, ml(m, n, bcs)) }, bcs, beta, unsafe { sm(c, ml(m, n, ccs)) }, ccs).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn ssyrk(alpha: f32, n: usize, k: usize, a: *const f32, acs: usize, beta: f32, c: *mut f32, ccs: usize, upper: u32) {
	L3::ssyrk(uplo(upper != 0), Trans::NoTrans, n, k, alpha, unsafe { s(a, ml(n, k, acs)) }, acs, beta, unsafe { sm(c, ml(n, n, ccs)) }, ccs, &mut vec![0.0f32; L3::ssyrk_work(n, k)]).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn ssyr2k(alpha: f32, n: usize, k: usize, a: *const f32, acs: usize, b: *const f32, bcs: usize, beta: f32, c: *mut f32, ccs: usize, upper: u32) {
	L3::ssyr2k(uplo(upper != 0), Trans::NoTrans, n, k, alpha, unsafe { s(a, ml(n, k, acs)) }, acs, unsafe { s(b, ml(n, k, bcs)) }, bcs, beta, unsafe { sm(c, ml(n, n, ccs)) }, ccs, &mut vec![0.0f32; L3::ssyr2k_work(n, k)]).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn strmm_left(alpha: f32, m: usize, n: usize, a: *const f32, acs: usize, upper: u32, unit: u32, b: *mut f32, bcs: usize) {
	L3::strmm(Side::Left, uplo(upper != 0), Trans::NoTrans, diag(unit != 0), m, n, alpha, unsafe { s(a, ml(m, m, acs)) }, acs, unsafe { sm(b, ml(m, n, bcs)) }, bcs, &mut vec![0.0f32; L3::strmm_work(Side::Left, Trans::NoTrans, m, n)]).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn strmm_right(alpha: f32, m: usize, n: usize, a: *const f32, acs: usize, upper: u32, unit: u32, b: *mut f32, bcs: usize) {
	L3::strmm(Side::Right, uplo(upper != 0), Trans::NoTrans, diag(unit != 0), m, n, alpha, unsafe { s(a, ml(n, n, acs)) }, acs, unsafe { sm(b, ml(m, n, bcs)) }, bcs, &mut vec![0.0f32; L3::strmm_work(Side::Right, Trans::NoTrans, m, n)]).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn strsm_left(alpha: f32, m: usize, n: usize, a: *const f32, acs: usize, upper: u32, unit: u32, b: *mut f32, bcs: usize) {
	L3::strsm(Side::Left, uplo(upper != 0), Trans::NoTrans, diag(unit != 0), m, n, alpha, unsafe { s(a, ml(m, m, acs)) }, acs, unsafe { sm(b, ml(m, n, bcs)) }, bcs, &mut vec![0.0f32; L3::strsm_work(Side::Left, Trans::NoTrans, m, n)]).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn strsm_right(alpha: f32, m: usize, n: usize, a: *const f32, acs: usize, upper: u32, unit: u32, b: *mut f32, bcs: usize) {
	L3::strsm(Side::Right, uplo(upper != 0), Trans::NoTrans, diag(unit != 0), m, n, alpha, unsafe { s(a, ml(n, n, acs)) }, acs, unsafe { sm(b, ml(m, n, bcs)) }, bcs, &mut vec![0.0f32; L3::strsm_work(Side::Right, Trans::NoTrans, m, n)]).unwrap();
}

// ---------------- z-routines (C64) ----------------

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn zcopy(x: *const C64, y: *mut C64, n: usize) {
	L1::zcopy(unsafe { s(x, n) }, unsafe { sm(y, n) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn zswap(x: *mut C64, y: *mut C64, n: usize) {
	L1::zswap(unsafe { sm(x, n) }, unsafe { sm(y, n) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn zscal(are: f64, aim: f64, x: *mut C64, n: usize) {
	L1::zscal(C64::new(are, aim), unsafe { sm(x, n) });
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn zdscal(alpha: f64, x: *mut C64, n: usize) {
	L1::zdscal(alpha, unsafe { sm(x, n) });
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn zaxpy(are: f64, aim: f64, x: *const C64, y: *mut C64, n: usize) {
	L1::zaxpy(C64::new(are, aim), unsafe { s(x, n) }, unsafe { sm(y, n) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn zdrot(x: *mut C64, y: *mut C64, c: f64, s_: f64, n: usize) {
	L1::zdrot(unsafe { sm(x, n) }, unsafe { sm(y, n) }, c, s_).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn zdotu(x: *const C64, y: *const C64, n: usize, out: *mut C64) {
	let v = L1::zdotu(unsafe { s(x, n) }, unsafe { s(y, n) }).unwrap();
	unsafe { *out = v };
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn zdotc(x: *const C64, y: *const C64, n: usize, out: *mut C64) {
	let v = L1::zdotc(unsafe { s(x, n) }, unsafe { s(y, n) }).unwrap();
	unsafe { *out = v };
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn dznrm2(x: *const C64, n: usize) -> f64 {
	L1::dznrm2(unsafe { s(x, n) })
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn dzasum(x: *const C64, n: usize) -> f64 {
	L1::dzasum(unsafe { s(x, n) })
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn izamax(x: *const C64, n: usize) -> usize {
	L1::izamax(unsafe { s(x, n) })
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn zrotg(are: f64, aim: f64, bre: f64, bim: f64, c: *mut f64, s_: *mut C64, r: *mut C64) {
	let g = L1::zrotg(C64::new(are, aim), C64::new(bre, bim));
	unsafe {
		*c = g.c;
		*s_ = g.s;
		*r = g.r;
	}
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn zgemv(are: f64, aim: f64, nrows: usize, ncols: usize, a: *const C64, cs: usize, x: *const C64, bre: f64, bim: f64, y: *mut C64) {
	L2::zgemv(Trans::NoTrans, nrows, ncols, C64::new(are, aim), unsafe { s(a, ml(nrows, ncols, cs)) }, cs, unsafe { s(x, ncols) }, C64::new(bre, bim), unsafe { sm(y, nrows) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn zgemv_t(are: f64, aim: f64, nrows: usize, ncols: usize, a: *const C64, cs: usize, x: *const C64, bre: f64, bim: f64, y: *mut C64) {
	L2::zgemv(Trans::Trans, nrows, ncols, C64::new(are, aim), unsafe { s(a, ml(nrows, ncols, cs)) }, cs, unsafe { s(x, nrows) }, C64::new(bre, bim), unsafe { sm(y, ncols) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn zgemv_c(are: f64, aim: f64, nrows: usize, ncols: usize, a: *const C64, cs: usize, x: *const C64, bre: f64, bim: f64, y: *mut C64) {
	L2::zgemv(Trans::ConjTrans, nrows, ncols, C64::new(are, aim), unsafe { s(a, ml(nrows, ncols, cs)) }, cs, unsafe { s(x, nrows) }, C64::new(bre, bim), unsafe { sm(y, ncols) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn zgeru(are: f64, aim: f64, nrows: usize, ncols: usize, a: *mut C64, cs: usize, x: *const C64, y: *const C64) {
	L2::zgeru(nrows, ncols, C64::new(are, aim), unsafe { s(x, nrows) }, unsafe { s(y, ncols) }, unsafe { sm(a, ml(nrows, ncols, cs)) }, cs).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn zgerc(are: f64, aim: f64, nrows: usize, ncols: usize, a: *mut C64, cs: usize, x: *const C64, y: *const C64) {
	L2::zgerc(nrows, ncols, C64::new(are, aim), unsafe { s(x, nrows) }, unsafe { s(y, ncols) }, unsafe { sm(a, ml(nrows, ncols, cs)) }, cs).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn zhemv(are: f64, aim: f64, n: usize, a: *const C64, cs: usize, upper: u32, x: *const C64, bre: f64, bim: f64, y: *mut C64) {
	L2::zhemv(uplo(upper != 0), n, C64::new(are, aim), unsafe { s(a, ml(n, n, cs)) }, cs, unsafe { s(x, n) }, C64::new(bre, bim), unsafe { sm(y, n) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn zher(alpha: f64, n: usize, a: *mut C64, cs: usize, upper: u32, x: *const C64) {
	L2::zher(uplo(upper != 0), n, alpha, unsafe { s(x, n) }, unsafe { sm(a, ml(n, n, cs)) }, cs).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn zher2(are: f64, aim: f64, n: usize, a: *mut C64, cs: usize, upper: u32, x: *const C64, y: *const C64) {
	L2::zher2(uplo(upper != 0), n, C64::new(are, aim), unsafe { s(x, n) }, unsafe { s(y, n) }, unsafe { sm(a, ml(n, n, cs)) }, cs).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn ztrmv(n: usize, a: *const C64, cs: usize, upper: u32, unit: u32, x: *mut C64) {
	L2::ztrmv(uplo(upper != 0), Trans::NoTrans, diag(unit != 0), n, unsafe { s(a, ml(n, n, cs)) }, cs, unsafe { sm(x, n) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn ztrsv(n: usize, a: *const C64, cs: usize, upper: u32, unit: u32, x: *mut C64) {
	L2::ztrsv(uplo(upper != 0), Trans::NoTrans, diag(unit != 0), n, unsafe { s(a, ml(n, n, cs)) }, cs, unsafe { sm(x, n) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn zgemm(are: f64, aim: f64, m: usize, k: usize, n: usize, a: *const C64, acs: usize, b: *const C64, bcs: usize, bre: f64, bim: f64, c: *mut C64, ccs: usize) {
	L3::zgemm(Trans::NoTrans, Trans::NoTrans, m, n, k, C64::new(are, aim), unsafe { s(a, ml(m, k, acs)) }, acs, unsafe { s(b, ml(k, n, bcs)) }, bcs, C64::new(bre, bim), unsafe { sm(c, ml(m, n, ccs)) }, ccs, &mut vec![C64::ZERO; L3::zgemm_work(Trans::NoTrans, Trans::NoTrans, m, n, k)]).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn zhemm_left(are: f64, aim: f64, m: usize, n: usize, a: *const C64, acs: usize, upper: u32, b: *const C64, bcs: usize, bre: f64, bim: f64, c: *mut C64, ccs: usize) {
	L3::zhemm(Side::Left, uplo(upper != 0), m, n, C64::new(are, aim), unsafe { s(a, ml(m, m, acs)) }, acs, unsafe { s(b, ml(m, n, bcs)) }, bcs, C64::new(bre, bim), unsafe { sm(c, ml(m, n, ccs)) }, ccs).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn zhemm_right(are: f64, aim: f64, m: usize, n: usize, a: *const C64, acs: usize, upper: u32, b: *const C64, bcs: usize, bre: f64, bim: f64, c: *mut C64, ccs: usize) {
	L3::zhemm(Side::Right, uplo(upper != 0), m, n, C64::new(are, aim), unsafe { s(a, ml(n, n, acs)) }, acs, unsafe { s(b, ml(m, n, bcs)) }, bcs, C64::new(bre, bim), unsafe { sm(c, ml(m, n, ccs)) }, ccs).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn zherk(alpha: f64, n: usize, k: usize, a: *const C64, acs: usize, beta: f64, c: *mut C64, ccs: usize, upper: u32) {
	L3::zherk(uplo(upper != 0), Trans::NoTrans, n, k, alpha, unsafe { s(a, ml(n, k, acs)) }, acs, beta, unsafe { sm(c, ml(n, n, ccs)) }, ccs, &mut vec![C64::ZERO; L3::zherk_work(n, k)]).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn zher2k(are: f64, aim: f64, n: usize, k: usize, a: *const C64, acs: usize, b: *const C64, bcs: usize, beta: f64, c: *mut C64, ccs: usize, upper: u32) {
	L3::zher2k(uplo(upper != 0), Trans::NoTrans, n, k, C64::new(are, aim), unsafe { s(a, ml(n, k, acs)) }, acs, unsafe { s(b, ml(n, k, bcs)) }, bcs, beta, unsafe { sm(c, ml(n, n, ccs)) }, ccs, &mut vec![C64::ZERO; L3::zher2k_work(n, k)]).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn ztrmm_left(are: f64, aim: f64, m: usize, n: usize, a: *const C64, acs: usize, upper: u32, unit: u32, b: *mut C64, bcs: usize) {
	L3::ztrmm(Side::Left, uplo(upper != 0), Trans::NoTrans, diag(unit != 0), m, n, C64::new(are, aim), unsafe { s(a, ml(m, m, acs)) }, acs, unsafe { sm(b, ml(m, n, bcs)) }, bcs, &mut vec![C64::ZERO; L3::ztrmm_work(Side::Left, Trans::NoTrans, m, n)]).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn ztrmm_right(are: f64, aim: f64, m: usize, n: usize, a: *const C64, acs: usize, upper: u32, unit: u32, b: *mut C64, bcs: usize) {
	L3::ztrmm(Side::Right, uplo(upper != 0), Trans::NoTrans, diag(unit != 0), m, n, C64::new(are, aim), unsafe { s(a, ml(n, n, acs)) }, acs, unsafe { sm(b, ml(m, n, bcs)) }, bcs, &mut vec![C64::ZERO; L3::ztrmm_work(Side::Right, Trans::NoTrans, m, n)]).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn ztrsm_left(are: f64, aim: f64, m: usize, n: usize, a: *const C64, acs: usize, upper: u32, unit: u32, b: *mut C64, bcs: usize) {
	L3::ztrsm(Side::Left, uplo(upper != 0), Trans::NoTrans, diag(unit != 0), m, n, C64::new(are, aim), unsafe { s(a, ml(m, m, acs)) }, acs, unsafe { sm(b, ml(m, n, bcs)) }, bcs, &mut vec![C64::ZERO; L3::ztrsm_work(Side::Left, Trans::NoTrans, m, n)]).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn ztrsm_right(are: f64, aim: f64, m: usize, n: usize, a: *const C64, acs: usize, upper: u32, unit: u32, b: *mut C64, bcs: usize) {
	L3::ztrsm(Side::Right, uplo(upper != 0), Trans::NoTrans, diag(unit != 0), m, n, C64::new(are, aim), unsafe { s(a, ml(n, n, acs)) }, acs, unsafe { sm(b, ml(m, n, bcs)) }, bcs, &mut vec![C64::ZERO; L3::ztrsm_work(Side::Right, Trans::NoTrans, m, n)]).unwrap();
}

// ---------------- c-routines (C32) ----------------

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn ccopy(x: *const C32, y: *mut C32, n: usize) {
	L1::ccopy(unsafe { s(x, n) }, unsafe { sm(y, n) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn cswap(x: *mut C32, y: *mut C32, n: usize) {
	L1::cswap(unsafe { sm(x, n) }, unsafe { sm(y, n) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn cscal(are: f32, aim: f32, x: *mut C32, n: usize) {
	L1::cscal(C32::new(are, aim), unsafe { sm(x, n) });
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn csscal(alpha: f32, x: *mut C32, n: usize) {
	L1::csscal(alpha, unsafe { sm(x, n) });
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn caxpy(are: f32, aim: f32, x: *const C32, y: *mut C32, n: usize) {
	L1::caxpy(C32::new(are, aim), unsafe { s(x, n) }, unsafe { sm(y, n) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn csrot(x: *mut C32, y: *mut C32, c: f32, s_: f32, n: usize) {
	L1::csrot(unsafe { sm(x, n) }, unsafe { sm(y, n) }, c, s_).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn cdotu(x: *const C32, y: *const C32, n: usize, out: *mut C32) {
	let v = L1::cdotu(unsafe { s(x, n) }, unsafe { s(y, n) }).unwrap();
	unsafe { *out = v };
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn cdotc(x: *const C32, y: *const C32, n: usize, out: *mut C32) {
	let v = L1::cdotc(unsafe { s(x, n) }, unsafe { s(y, n) }).unwrap();
	unsafe { *out = v };
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn scnrm2(x: *const C32, n: usize) -> f32 {
	L1::scnrm2(unsafe { s(x, n) })
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn scasum(x: *const C32, n: usize) -> f32 {
	L1::scasum(unsafe { s(x, n) })
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn icamax(x: *const C32, n: usize) -> usize {
	L1::icamax(unsafe { s(x, n) })
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn crotg(are: f32, aim: f32, bre: f32, bim: f32, c: *mut f32, s_: *mut C32, r: *mut C32) {
	let g = L1::crotg(C32::new(are, aim), C32::new(bre, bim));
	unsafe {
		*c = g.c;
		*s_ = g.s;
		*r = g.r;
	}
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn cgemv(are: f32, aim: f32, nrows: usize, ncols: usize, a: *const C32, cs: usize, x: *const C32, bre: f32, bim: f32, y: *mut C32) {
	L2::cgemv(Trans::NoTrans, nrows, ncols, C32::new(are, aim), unsafe { s(a, ml(nrows, ncols, cs)) }, cs, unsafe { s(x, ncols) }, C32::new(bre, bim), unsafe { sm(y, nrows) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn cgemv_t(are: f32, aim: f32, nrows: usize, ncols: usize, a: *const C32, cs: usize, x: *const C32, bre: f32, bim: f32, y: *mut C32) {
	L2::cgemv(Trans::Trans, nrows, ncols, C32::new(are, aim), unsafe { s(a, ml(nrows, ncols, cs)) }, cs, unsafe { s(x, nrows) }, C32::new(bre, bim), unsafe { sm(y, ncols) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn cgemv_c(are: f32, aim: f32, nrows: usize, ncols: usize, a: *const C32, cs: usize, x: *const C32, bre: f32, bim: f32, y: *mut C32) {
	L2::cgemv(Trans::ConjTrans, nrows, ncols, C32::new(are, aim), unsafe { s(a, ml(nrows, ncols, cs)) }, cs, unsafe { s(x, nrows) }, C32::new(bre, bim), unsafe { sm(y, ncols) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn cgeru(are: f32, aim: f32, nrows: usize, ncols: usize, a: *mut C32, cs: usize, x: *const C32, y: *const C32) {
	L2::cgeru(nrows, ncols, C32::new(are, aim), unsafe { s(x, nrows) }, unsafe { s(y, ncols) }, unsafe { sm(a, ml(nrows, ncols, cs)) }, cs).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn cgerc(are: f32, aim: f32, nrows: usize, ncols: usize, a: *mut C32, cs: usize, x: *const C32, y: *const C32) {
	L2::cgerc(nrows, ncols, C32::new(are, aim), unsafe { s(x, nrows) }, unsafe { s(y, ncols) }, unsafe { sm(a, ml(nrows, ncols, cs)) }, cs).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn chemv(are: f32, aim: f32, n: usize, a: *const C32, cs: usize, upper: u32, x: *const C32, bre: f32, bim: f32, y: *mut C32) {
	L2::chemv(uplo(upper != 0), n, C32::new(are, aim), unsafe { s(a, ml(n, n, cs)) }, cs, unsafe { s(x, n) }, C32::new(bre, bim), unsafe { sm(y, n) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn cher(alpha: f32, n: usize, a: *mut C32, cs: usize, upper: u32, x: *const C32) {
	L2::cher(uplo(upper != 0), n, alpha, unsafe { s(x, n) }, unsafe { sm(a, ml(n, n, cs)) }, cs).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn cher2(are: f32, aim: f32, n: usize, a: *mut C32, cs: usize, upper: u32, x: *const C32, y: *const C32) {
	L2::cher2(uplo(upper != 0), n, C32::new(are, aim), unsafe { s(x, n) }, unsafe { s(y, n) }, unsafe { sm(a, ml(n, n, cs)) }, cs).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn ctrmv(n: usize, a: *const C32, cs: usize, upper: u32, unit: u32, x: *mut C32) {
	L2::ctrmv(uplo(upper != 0), Trans::NoTrans, diag(unit != 0), n, unsafe { s(a, ml(n, n, cs)) }, cs, unsafe { sm(x, n) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn ctrsv(n: usize, a: *const C32, cs: usize, upper: u32, unit: u32, x: *mut C32) {
	L2::ctrsv(uplo(upper != 0), Trans::NoTrans, diag(unit != 0), n, unsafe { s(a, ml(n, n, cs)) }, cs, unsafe { sm(x, n) }).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn cgemm(are: f32, aim: f32, m: usize, k: usize, n: usize, a: *const C32, acs: usize, b: *const C32, bcs: usize, bre: f32, bim: f32, c: *mut C32, ccs: usize) {
	L3::cgemm(Trans::NoTrans, Trans::NoTrans, m, n, k, C32::new(are, aim), unsafe { s(a, ml(m, k, acs)) }, acs, unsafe { s(b, ml(k, n, bcs)) }, bcs, C32::new(bre, bim), unsafe { sm(c, ml(m, n, ccs)) }, ccs, &mut vec![C32::ZERO; L3::cgemm_work(Trans::NoTrans, Trans::NoTrans, m, n, k)]).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn chemm_left(are: f32, aim: f32, m: usize, n: usize, a: *const C32, acs: usize, upper: u32, b: *const C32, bcs: usize, bre: f32, bim: f32, c: *mut C32, ccs: usize) {
	L3::chemm(Side::Left, uplo(upper != 0), m, n, C32::new(are, aim), unsafe { s(a, ml(m, m, acs)) }, acs, unsafe { s(b, ml(m, n, bcs)) }, bcs, C32::new(bre, bim), unsafe { sm(c, ml(m, n, ccs)) }, ccs).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn chemm_right(are: f32, aim: f32, m: usize, n: usize, a: *const C32, acs: usize, upper: u32, b: *const C32, bcs: usize, bre: f32, bim: f32, c: *mut C32, ccs: usize) {
	L3::chemm(Side::Right, uplo(upper != 0), m, n, C32::new(are, aim), unsafe { s(a, ml(n, n, acs)) }, acs, unsafe { s(b, ml(m, n, bcs)) }, bcs, C32::new(bre, bim), unsafe { sm(c, ml(m, n, ccs)) }, ccs).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn cherk(alpha: f32, n: usize, k: usize, a: *const C32, acs: usize, beta: f32, c: *mut C32, ccs: usize, upper: u32) {
	L3::cherk(uplo(upper != 0), Trans::NoTrans, n, k, alpha, unsafe { s(a, ml(n, k, acs)) }, acs, beta, unsafe { sm(c, ml(n, n, ccs)) }, ccs, &mut vec![C32::ZERO; L3::cherk_work(n, k)]).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn cher2k(are: f32, aim: f32, n: usize, k: usize, a: *const C32, acs: usize, b: *const C32, bcs: usize, beta: f32, c: *mut C32, ccs: usize, upper: u32) {
	L3::cher2k(uplo(upper != 0), Trans::NoTrans, n, k, C32::new(are, aim), unsafe { s(a, ml(n, k, acs)) }, acs, unsafe { s(b, ml(n, k, bcs)) }, bcs, beta, unsafe { sm(c, ml(n, n, ccs)) }, ccs, &mut vec![C32::ZERO; L3::cher2k_work(n, k)]).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn ctrmm_left(are: f32, aim: f32, m: usize, n: usize, a: *const C32, acs: usize, upper: u32, unit: u32, b: *mut C32, bcs: usize) {
	L3::ctrmm(Side::Left, uplo(upper != 0), Trans::NoTrans, diag(unit != 0), m, n, C32::new(are, aim), unsafe { s(a, ml(m, m, acs)) }, acs, unsafe { sm(b, ml(m, n, bcs)) }, bcs, &mut vec![C32::ZERO; L3::ctrmm_work(Side::Left, Trans::NoTrans, m, n)]).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn ctrmm_right(are: f32, aim: f32, m: usize, n: usize, a: *const C32, acs: usize, upper: u32, unit: u32, b: *mut C32, bcs: usize) {
	L3::ctrmm(Side::Right, uplo(upper != 0), Trans::NoTrans, diag(unit != 0), m, n, C32::new(are, aim), unsafe { s(a, ml(n, n, acs)) }, acs, unsafe { sm(b, ml(m, n, bcs)) }, bcs, &mut vec![C32::ZERO; L3::ctrmm_work(Side::Right, Trans::NoTrans, m, n)]).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn ctrsm_left(are: f32, aim: f32, m: usize, n: usize, a: *const C32, acs: usize, upper: u32, unit: u32, b: *mut C32, bcs: usize) {
	L3::ctrsm(Side::Left, uplo(upper != 0), Trans::NoTrans, diag(unit != 0), m, n, C32::new(are, aim), unsafe { s(a, ml(m, m, acs)) }, acs, unsafe { sm(b, ml(m, n, bcs)) }, bcs, &mut vec![C32::ZERO; L3::ctrsm_work(Side::Left, Trans::NoTrans, m, n)]).unwrap();
}

/// # Safety
/// Pointers must be valid for the extents given by the module
/// conventions (matrices: `cs*(cols-1)+rows` elements).
#[no_mangle]
pub unsafe extern "C" fn ctrsm_right(are: f32, aim: f32, m: usize, n: usize, a: *const C32, acs: usize, upper: u32, unit: u32, b: *mut C32, bcs: usize) {
	L3::ctrsm(Side::Right, uplo(upper != 0), Trans::NoTrans, diag(unit != 0), m, n, C32::new(are, aim), unsafe { s(a, ml(n, n, acs)) }, acs, unsafe { sm(b, ml(m, n, bcs)) }, bcs, &mut vec![C32::ZERO; L3::ctrsm_work(Side::Right, Trans::NoTrans, m, n)]).unwrap();
}
