//! blas-bench — the BLAS layer's own measurement harness: the roofline
//! rows, the cross-target determinism probes, and the machine-ceiling
//! probes, for all four number types (f64/f32/c64/c32). Self-contained:
//! depends only on blas-wasm, so it builds in seconds and the
//! repository is the complete product — code (../src), correctness
//! (../tests), measurement (here). Scripts: `l{1,2,3}-roofline.mjs` in
//! this folder; the results scoreboard is this folder's README.md; the
//! expected probe bit patterns are ../tests/README.md.
//!
//! State is plain column-major Vecs with cs = n (no padding). The
//! determinism probes are self-contained (own LCG, own matrices) and
//! reproduce the probe bit patterns recorded in ../tests/README.md
//! exactly. The faer market races were run in the predecessor
//! repository and are not reproducible from this tree (STATUS.md).

struct State {
    n: usize,
    a: Vec<f64>,
    b: Vec<f64>,
    // sacrificial destination (triad target + L2/L3 mutation target)
    sym: Vec<f64>,
    // a's values with a dominant diagonal: solves stay bounded
    tri: Vec<f64>,
    rhs: Vec<f64>,
    a32: Vec<f32>,
    b32: Vec<f32>,
    sym32: Vec<f32>,
    tri32: Vec<f32>,
    rhs32: Vec<f32>,
    // c64 twins — own LCG fills (re/im interleaved draws), same roles
    az: Vec<C64>,
    bz: Vec<C64>,
    symz: Vec<C64>,
    triz: Vec<C64>,
    rhsz: Vec<C64>,
    // c32 twins — the c64 fills cast to f32, same roles
    ac: Vec<C32>,
    bc: Vec<C32>,
    symc: Vec<C32>,
    tric: Vec<C32>,
    rhsc: Vec<C32>,
    // caller-side workspaces for the races, grown on the warmup call so
    // no allocation or zero-fill lands inside a timed call
    work: Vec<f64>,
    work32: Vec<f32>,
    workz: Vec<C64>,
    workc: Vec<C32>,
}

/// The race workspace of at least `need` elements, grown (outside the
/// timed region: the scripts warm each candidate up first) and sliced.
fn ws<T: Copy>(v: &mut Vec<T>, need: usize, zero: T) -> &mut [T] {
    if v.len() < need {
        v.resize(need, zero);
    }
    &mut v[..need]
}

use blas_wasm::C64;

use blas_wasm::{Diag, Side, Trans, Uplo};

struct StateCell(core::cell::UnsafeCell<Option<State>>);
unsafe impl Sync for StateCell {}
static STATE: StateCell = StateCell(core::cell::UnsafeCell::new(None));

fn state_mut() -> &'static mut State {
    unsafe { (*STATE.0.get()).as_mut().expect("call setup(n) first") }
}

// deterministic fill (splitmix-style LCG), values in [-1, 1] —
// same recipe and seeds as the predecessor harness, so the probe bit
// patterns carry over
fn fill(n: usize, mut s: u64) -> Vec<f64> {
    let mut v = vec![0.0f64; n * n];
    for j in 0..n {
        for i in 0..n {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            v[j * n + i] = ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
        }
    }
    v
}

/// Column-major n×n fill from a probe's LCG closure, column by column
/// (the probe bit patterns depend on this fill order).
fn alloc_mat_f64(n: usize, next: &mut impl FnMut() -> f64) -> Vec<f64> {
    let mut v = vec![0.0f64; n * n];
    for j in 0..n {
        for i in 0..n {
            v[j * n + i] = next();
        }
    }
    v
}

fn alloc_mat_f32(n: usize, next: &mut impl FnMut() -> f32) -> Vec<f32> {
    let mut v = vec![0.0f32; n * n];
    for j in 0..n {
        for i in 0..n {
            v[j * n + i] = next();
        }
    }
    v
}

#[no_mangle]
pub extern "C" fn setup(n: usize) {
    let a = fill(n, 0x9E3779B97F4A7C15);
    let b = fill(n, 0xD1B54A32D192ED03);
    // symmetric, diagonally dominant (same construction as the
    // predecessor harness: a + aᵀ with a boosted diagonal)
    let mut sym = vec![0.0f64; n * n];
    for j in 0..n {
        for i in 0..n {
            sym[j * n + i] = a[j * n + i] + a[i * n + j];
        }
    }
    for i in 0..n {
        sym[i * n + i] += 2.0 * n as f64;
    }
    let mut tri = a.clone();
    for i in 0..n {
        tri[i * n + i] = 2.0 * n as f64 + 1.0;
    }
    let mut rhs = vec![0.0f64; n.max(1)];
    {
        let mut s = 0x853C49E6748FEA9Bu64;
        for v in rhs.iter_mut().take(n) {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *v = ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0;
        }
    }
    let a32: Vec<f32> = a.iter().map(|&v| v as f32).collect();
    let b32: Vec<f32> = b.iter().map(|&v| v as f32).collect();
    let sym32: Vec<f32> = sym.iter().map(|&v| v as f32).collect();
    let tri32: Vec<f32> = tri.iter().map(|&v| v as f32).collect();
    let rhs32: Vec<f32> = rhs.iter().map(|&v| v as f32).collect();
    let az = fill_z(n * n, 0x1B873593CC9E2D51);
    let bz = fill_z(n * n, 0xE6546B64B432C925);
    let symz = bz.clone(); // sacrificial destination, like sym
    let mut triz = az.clone();
    for i in 0..n {
        triz[i * n + i] = C64::new(2.0 * n as f64 + 1.0, 0.7);
    }
    let rhsz = fill_z(n.max(1), 0x2545F4914F6CDD1D);
    let cast = |v: &[C64]| -> Vec<C32> {
        v.iter().map(|z| C32::new(z.re as f32, z.im as f32)).collect()
    };
    let ac = cast(&az);
    let bc = cast(&bz);
    let symc = bc.clone(); // sacrificial destination
    let mut tric = ac.clone();
    for i in 0..n {
        tric[i * n + i] = C32::new(2.0 * n as f32 + 1.0, 0.7);
    }
    let rhsc = cast(&rhsz);
    unsafe {
        *STATE.0.get() = Some(State {
            n,
            a,
            b,
            sym,
            tri,
            rhs,
            a32,
            b32,
            sym32,
            tri32,
            rhs32,
            az,
            bz,
            symz,
            triz,
            rhsz,
            ac,
            bc,
            symc,
            tric,
            rhsc,
            work: Vec::new(),
            work32: Vec::new(),
            workz: Vec::new(),
            workc: Vec::new(),
        })
    }
}

/// Deterministic complex fill: re then im from one LCG stream.
fn fill_z(len: usize, mut s: u64) -> Vec<C64> {
    let mut next = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    (0..len)
        .map(|_| {
            let re = next();
            let im = next();
            C64::new(re, im)
        })
        .collect()
}

/// c32 twin of `alloc_mat_c64` (f32 closure).
fn alloc_mat_c32(n: usize, next: &mut impl FnMut() -> f32) -> Vec<blas_wasm::C32> {
    use blas_wasm::C32;
    let mut v = vec![C32::ZERO; n * n];
    for j in 0..n {
        for i in 0..n {
            let re = next();
            let im = next();
            v[j * n + i] = C32::new(re, im);
        }
    }
    v
}

/// Column-major n×n complex fill from a probe's LCG closure (re then
/// im per element, column-major order — probe bits depend on it).
fn alloc_mat_c64(n: usize, next: &mut impl FnMut() -> f64) -> Vec<C64> {
    let mut v = vec![C64::ZERO; n * n];
    for j in 0..n {
        for i in 0..n {
            let re = next();
            let im = next();
            v[j * n + i] = C64::new(re, im);
        }
    }
    v
}

#[no_mangle]
pub extern "C" fn run_l1_layer(op: usize) -> f64 {
    use blas_wasm::L1 as l1;
    let s = state_mut();
    let n = s.n;
    let acs = s.n;
    let bcs = s.n;
    let ap = s.a.as_mut_ptr();
    let bp = s.b.as_mut_ptr();
    let col = |p: *mut f64, cs: usize, j: usize| unsafe {
        core::slice::from_raw_parts_mut(p.add(j * cs), n)
    };
    let mut sink = 0.0f64;
    match op {
        0 => {
            for j in 0..n {
                l1::dcopy(col(ap, acs, j), col(bp, bcs, j)).unwrap();
            }
            sink += unsafe { *bp };
        }
        1 => {
            for j in 0..n {
                l1::dswap(col(ap, acs, j), col(bp, bcs, j)).unwrap();
            }
            sink += unsafe { *ap };
        }
        2 => {
            for j in 0..n {
                l1::dscal(-1.0, col(ap, acs, j));
            }
            sink += unsafe { *ap };
        }
        3 => {
            for j in 0..n {
                l1::daxpy(0.001, col(ap, acs, j), col(bp, bcs, j)).unwrap();
            }
            sink += unsafe { *bp };
        }
        4 => {
            for j in 0..n {
                l1::drot(col(ap, acs, j), col(bp, bcs, j), 0.8, 0.6).unwrap();
            }
            sink += unsafe { *ap };
        }
        5 => {
            for j in 0..n {
                sink += l1::ddot(col(ap, acs, j), col(bp, bcs, j)).unwrap();
            }
        }
        6 => {
            for j in 0..n {
                sink += l1::dnrm2(col(ap, acs, j));
            }
        }
        7 => {
            for j in 0..n {
                sink += l1::dasum(col(ap, acs, j));
            }
        }
        8 => {
            for j in 0..n {
                sink += l1::idamax(col(ap, acs, j)) as f64;
            }
        }
        _ => return f64::NAN,
    }
    sink
}


/// Level-2 roofline rows over the persistent state. op: 0 gemv,
/// 1 gemv_t, 2 ger, 3 symv, 4 trmv, 5 trsv, 6 syr, 7 syr2. Mutating
/// targets and constants chosen so values stay bounded across bench
/// iterations (small alpha on accumulating updates; trmv/trsv copy a
/// fresh x from b's first column each call — 8n bytes, noise next to
/// the 4·n²–16·n² the matrix stream moves).
#[no_mangle]
pub extern "C" fn run_l2_layer(op: usize) -> f64 {
    use blas_wasm::L1 as l1;
    use blas_wasm::L2 as l2;
    let s = state_mut();
    let n = s.n;
    let acs = s.n;
    let scs = s.n;
    let tcs = s.n;
    let a_len = if n == 0 { 0 } else { acs * (n - 1) + n };
    let s_len = if n == 0 { 0 } else { scs * (n - 1) + n };
    let t_len = if n == 0 { 0 } else { tcs * (n - 1) + n };
    let a = unsafe { core::slice::from_raw_parts(s.a.as_ptr(), a_len) };
    let bx = unsafe { core::slice::from_raw_parts(s.b.as_ptr(), n) }; // b's first column
    let sym = unsafe { core::slice::from_raw_parts_mut(s.sym.as_mut_ptr(), s_len) };
    let tri = unsafe { core::slice::from_raw_parts(s.tri.as_ptr(), t_len) };
    let y = unsafe { core::slice::from_raw_parts_mut(s.rhs.as_mut_ptr(), n) };
    match op {
        0 => l2::dgemv(Trans::NoTrans, n, n, 0.001, a, acs, bx, 0.5, y).unwrap(),
        1 => l2::dgemv(Trans::Trans, n, n, 0.001, a, acs, bx, 0.5, y).unwrap(),
        2 => l2::dger(n, n, 0.0001, bx, bx, sym, scs).unwrap(),
        3 => l2::dsymv(Uplo::Upper, n, 0.001, sym, scs, bx, 0.5, y).unwrap(),
        4 => {
            l1::dcopy(bx, y).unwrap();
            l2::dtrmv(Uplo::Lower, Trans::NoTrans, Diag::NonUnit, n, tri, tcs, y).unwrap();
        }
        5 => {
            l1::dcopy(bx, y).unwrap();
            l2::dtrsv(Uplo::Lower, Trans::NoTrans, Diag::NonUnit, n, tri, tcs, y).unwrap();
        }
        6 => l2::dsyr(Uplo::Upper, n, 0.0001, bx, sym, scs).unwrap(),
        7 => l2::dsyr2(Uplo::Upper, n, 0.0001, bx, bx, sym, scs).unwrap(),
        _ => return f64::NAN,
    }
    y[0] + sym[0] + y[n - 1]
}

/// Level-2 cross-target determinism probes: fixed LCG-filled 257×257
/// matrix (odd size — tails everywhere), one op each, folded to one
/// value with the layer's own asum. op: 0 gemv, 1 gemv_t, 2 ger,
/// 3 symv, 4 trmv, 5 trsv, 6 syr, 7 syr2.
#[no_mangle]
pub extern "C" fn run_l2_probe(op: usize) -> f64 {
    use blas_wasm::L1 as l1;
    use blas_wasm::L2 as l2;
    const N: usize = 257;
    let mut s = 7u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut a = alloc_mat_f64(N, &mut next);
    for i in 0..N {
        a[i * N + i] = 2.0 * N as f64 + 1.0; // solve-safe diagonal
    }
    let cs = N;
    let a_len = cs * (N - 1) + N;
    let x: [f64; N] = core::array::from_fn(|_| next());
    let mut y: [f64; N] = core::array::from_fn(|_| next());
    let am = &mut a[..a_len];
    match op {
        0 => l2::dgemv(Trans::NoTrans, N, N, 0.7, am, cs, &x, 0.3, &mut y).unwrap(),
        1 => l2::dgemv(Trans::Trans, N, N, 0.7, am, cs, &x, 0.3, &mut y).unwrap(),
        2 => l2::dger(N, N, 0.7, &x, &x, am, cs).unwrap(),
        3 => l2::dsymv(Uplo::Upper, N, 0.7, am, cs, &x, 0.3, &mut y).unwrap(),
        4 => {
            y.copy_from_slice(&x);
            l2::dtrmv(Uplo::Lower, Trans::NoTrans, Diag::NonUnit, N, am, cs, &mut y).unwrap();
        }
        5 => {
            y.copy_from_slice(&x);
            l2::dtrsv(Uplo::Lower, Trans::NoTrans, Diag::NonUnit, N, am, cs, &mut y).unwrap();
        }
        6 => l2::dsyr(Uplo::Upper, N, 0.7, &x, am, cs).unwrap(),
        7 => l2::dsyr2(Uplo::Upper, N, 0.7, &x, &x, am, cs).unwrap(),
        8 => {
            y.copy_from_slice(&x);
            l2::dtrmv(Uplo::Lower, Trans::Trans, Diag::NonUnit, N, am, cs, &mut y).unwrap();
        }
        9 => {
            y.copy_from_slice(&x);
            l2::dtrsv(Uplo::Lower, Trans::Trans, Diag::NonUnit, N, am, cs, &mut y).unwrap();
        }
        _ => return f64::NAN,
    }
    let a_probe = &a[..N];
    l1::dasum(&y) + y[0] + y[N - 1] + l1::dasum(a_probe)
}


/// Level-3 roofline rows over the persistent state. op: 0 gemm,
/// 1 symm_left, 2 syrk, 3 syr2k, 4 trmm_left, 5 trsm_left,
/// 6 trmm_right, 7 trsm_right. The in-place triangular ops re-copy B
/// fresh from `b` each call (an O(n²) copy against O(n³) work), so
/// values never grow across iterations; accumulating ops use small
/// alpha with beta = 0.5. `sym` is the sacrificial destination
/// (already true for the triad probe); `tri` (dominant diagonal) keeps
/// the solves bounded.
#[no_mangle]
pub extern "C" fn run_l3_layer(op: usize) -> f64 {
    use blas_wasm::L1 as l1;
    use blas_wasm::L3 as l3;
    let s = state_mut();
    let n = s.n;
    let acs = s.n;
    let bcs = s.n;
    let scs = s.n;
    let tcs = s.n;
    let len = |cs: usize| if n == 0 { 0 } else { cs * (n - 1) + n };
    let a = unsafe { core::slice::from_raw_parts(s.a.as_ptr(), len(acs)) };
    let b = unsafe { core::slice::from_raw_parts(s.b.as_ptr(), len(bcs)) };
    let tri = unsafe { core::slice::from_raw_parts(s.tri.as_ptr(), len(tcs)) };
    let sym = unsafe { core::slice::from_raw_parts_mut(s.sym.as_mut_ptr(), len(scs)) };
    let refresh = |dst: &mut [f64]| {
        for j in 0..n {
            l1::dcopy(&b[j * bcs..j * bcs + n], &mut dst[j * scs..j * scs + n]).unwrap();
        }
    };
    match op {
        0 => l3::dgemm(Trans::NoTrans, Trans::NoTrans, n, n, n, 0.001, a, acs, b, bcs, 0.5, sym, scs, &mut vec![0.0f64; l3::dgemm_work(Trans::NoTrans, Trans::NoTrans, n, n, n)]).unwrap(),
        1 => l3::dsymm(Side::Left, Uplo::Upper, n, n, 0.001, tri, tcs, b, bcs, 0.5, sym, scs).unwrap(),
        2 => l3::dsyrk(Uplo::Upper, Trans::NoTrans, n, n, 0.001, a, acs, 0.5, sym, scs, &mut vec![0.0f64; l3::dsyrk_work(Trans::NoTrans, n, n)]).unwrap(),
        3 => l3::dsyr2k(Uplo::Upper, Trans::NoTrans, n, n, 0.001, a, acs, b, bcs, 0.5, sym, scs, &mut vec![0.0f64; l3::dsyr2k_work(Trans::NoTrans, n, n)]).unwrap(),
        4 => {
            refresh(sym);
            l3::dtrmm(Side::Left, Uplo::Lower, Trans::NoTrans, Diag::NonUnit, n, n, 1.0, tri, tcs, sym, scs, &mut vec![0.0f64; l3::dtrmm_work(Side::Left, Trans::NoTrans, n, n)]).unwrap();
        }
        5 => {
            refresh(sym);
            l3::dtrsm(Side::Left, Uplo::Lower, Trans::NoTrans, Diag::NonUnit, n, n, 1.0, tri, tcs, sym, scs, &mut vec![0.0f64; l3::dtrsm_work(Side::Left, Trans::NoTrans, n, n)]).unwrap();
        }
        6 => {
            refresh(sym);
            l3::dtrmm(Side::Right, Uplo::Upper, Trans::NoTrans, Diag::NonUnit, n, n, 1.0, tri, tcs, sym, scs, &mut vec![0.0f64; l3::dtrmm_work(Side::Right, Trans::NoTrans, n, n)]).unwrap();
        }
        7 => {
            refresh(sym);
            l3::dtrsm(Side::Right, Uplo::Upper, Trans::NoTrans, Diag::NonUnit, n, n, 1.0, tri, tcs, sym, scs, &mut vec![0.0f64; l3::dtrsm_work(Side::Right, Trans::NoTrans, n, n)]).unwrap();
        }
        _ => return f64::NAN,
    }
    sym[0] + sym[len(scs) - 1]
}

/// Level-3 cross-target determinism probes: fixed LCG-filled 65×65
/// matrices (odd — tails everywhere), one op each, folded column-wise
/// with the layer's own asum. op order matches run_l3_layer plus
/// symm_right at 8.
#[no_mangle]
pub extern "C" fn run_l3_probe(op: usize) -> f64 {
    use blas_wasm::L1 as l1;
    use blas_wasm::L3 as l3;
    const N: usize = 65;
    const K: usize = 33;
    let mut st = 11u64;
    let mut next = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut a = alloc_mat_f64(N, &mut next);
    for i in 0..N {
        a[i * N + i] = 2.0 * N as f64 + 1.0;
    }
    let b = alloc_mat_f64(N, &mut next);
    let mut c = alloc_mat_f64(N, &mut next);
    let (acs, bcs, ccs) = (N, N, N);
    let len = |cs: usize| cs * (N - 1) + N;
    let av = &a[..len(acs)];
    let bv = &b[..len(bcs)];
    let cv = &mut c[..len(ccs)];
    match op {
        0 => l3::dgemm(Trans::NoTrans, Trans::NoTrans, N, N, K, 0.7, av, acs, bv, bcs, 0.3, cv, ccs, &mut vec![0.0f64; l3::dgemm_work(Trans::NoTrans, Trans::NoTrans, N, N, K)]).unwrap(),
        1 => l3::dsymm(Side::Left, Uplo::Upper, N, N, 0.7, av, acs, bv, bcs, 0.3, cv, ccs).unwrap(),
        2 => l3::dsyrk(Uplo::Upper, Trans::NoTrans, N, K, 0.7, av, acs, 0.3, cv, ccs, &mut vec![0.0f64; l3::dsyrk_work(Trans::NoTrans, N, K)]).unwrap(),
        3 => l3::dsyr2k(Uplo::Upper, Trans::NoTrans, N, K, 0.7, av, acs, bv, bcs, 0.3, cv, ccs, &mut vec![0.0f64; l3::dsyr2k_work(Trans::NoTrans, N, K)]).unwrap(),
        4 => l3::dtrmm(Side::Left, Uplo::Lower, Trans::NoTrans, Diag::NonUnit, N, N, 0.7, av, acs, cv, ccs, &mut vec![0.0f64; l3::dtrmm_work(Side::Left, Trans::NoTrans, N, N)]).unwrap(),
        5 => l3::dtrsm(Side::Left, Uplo::Lower, Trans::NoTrans, Diag::NonUnit, N, N, 0.7, av, acs, cv, ccs, &mut vec![0.0f64; l3::dtrsm_work(Side::Left, Trans::NoTrans, N, N)]).unwrap(),
        6 => l3::dtrmm(Side::Right, Uplo::Upper, Trans::NoTrans, Diag::NonUnit, N, N, 0.7, av, acs, cv, ccs, &mut vec![0.0f64; l3::dtrmm_work(Side::Right, Trans::NoTrans, N, N)]).unwrap(),
        7 => l3::dtrsm(Side::Right, Uplo::Upper, Trans::NoTrans, Diag::NonUnit, N, N, 0.7, av, acs, cv, ccs, &mut vec![0.0f64; l3::dtrsm_work(Side::Right, Trans::NoTrans, N, N)]).unwrap(),
        8 => l3::dsymm(Side::Right, Uplo::Upper, N, N, 0.7, av, acs, bv, bcs, 0.3, cv, ccs).unwrap(),
        // the transposed forms (todos S1): the copy zone, the packed
        // transposed read (a shape above this type's packed threshold,
        // its own fold), the syrk family, trmm and trsm on both sides
        9 => l3::dgemm(Trans::Trans, Trans::NoTrans, N, N, K, 0.7, av, acs, bv, bcs, 0.3, cv, ccs, &mut vec![0.0f64; l3::dgemm_work(Trans::Trans, Trans::NoTrans, N, N, K)]).unwrap(),
        10 => {
            const PK: usize = 520;
            let ab = alloc_mat_f64(PK, &mut next);
            let bb = alloc_mat_f64(PK, &mut next);
            let mut cb = alloc_mat_f64(PK, &mut next);
            let cb = &mut cb[..PK * 8];
            l3::dgemm(Trans::Trans, Trans::NoTrans, PK, 8, PK, 0.7, &ab, PK, &bb[..PK * 8], PK, 0.3, cb, PK, &mut vec![0.0f64; l3::dgemm_work(Trans::Trans, Trans::NoTrans, PK, 8, PK)]).unwrap();
            let mut fold = 0.0f64;
            for j in 0..8 {
                fold += l1::dasum(&cb[j * PK..j * PK + PK]);
            }
            return fold;
        }
        11 => l3::dsyrk(Uplo::Upper, Trans::Trans, N, K, 0.7, av, acs, 0.3, cv, ccs, &mut vec![0.0f64; l3::dsyrk_work(Trans::Trans, N, K)]).unwrap(),
        12 => l3::dtrmm(Side::Left, Uplo::Lower, Trans::Trans, Diag::NonUnit, N, N, 0.7, av, acs, cv, ccs, &mut vec![0.0f64; l3::dtrmm_work(Side::Left, Trans::Trans, N, N)]).unwrap(),
        13 => l3::dtrsm(Side::Left, Uplo::Lower, Trans::Trans, Diag::NonUnit, N, N, 0.7, av, acs, cv, ccs, &mut vec![0.0f64; l3::dtrsm_work(Side::Left, Trans::Trans, N, N)]).unwrap(),
        14 => l3::dtrmm(Side::Right, Uplo::Upper, Trans::Trans, Diag::NonUnit, N, N, 0.7, av, acs, cv, ccs, &mut vec![0.0f64; l3::dtrmm_work(Side::Right, Trans::Trans, N, N)]).unwrap(),
        15 => l3::dtrsm(Side::Right, Uplo::Upper, Trans::Trans, Diag::NonUnit, N, N, 0.7, av, acs, cv, ccs, &mut vec![0.0f64; l3::dtrsm_work(Side::Right, Trans::Trans, N, N)]).unwrap(),
        _ => return f64::NAN,
    }
    let mut fold = 0.0;
    for j in 0..N {
        fold += l1::dasum(&cv[j * ccs..j * ccs + N]);
    }
    fold + cv[0] + cv[len(ccs) - 1]
}


/// Cross-target determinism probe: fixed LCG data (len 1001 — odd, so the
/// scalar tail runs too), one reduction per op. Native bin and wasm build
/// must return identical bits (the lane-emulation construction in
/// src/lanes.rs). op: 0 dot, 1 asum, 2 nrm2, 3 iamax.
#[no_mangle]
pub extern "C" fn run_l1_probe(op: usize) -> f64 {
    use blas_wasm::L1 as l1;
    const LEN: usize = 1001;
    let mut x = [0.0f64; LEN];
    let mut y = [0.0f64; LEN];
    let mut s = 42u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    for v in x.iter_mut() {
        *v = next();
    }
    for v in y.iter_mut() {
        *v = next();
    }
    match op {
        0 => l1::ddot(&x, &y).unwrap(),
        1 => l1::dasum(&x),
        2 => l1::dnrm2(&x),
        3 => l1::idamax(&x) as f64,
        _ => f64::NAN,
    }
}
/// Memory-bandwidth probe: triad c = a + 2.5·b over the n×n state
/// matrices. Bytes moved per call = 3·8·n² (read a, read b, write c).
#[no_mangle]
pub extern "C" fn run_ceiling_bw() -> f64 {
    // the original version allocated + copied an n×n
    // matrix INSIDE the timed region (to_owned per call) and streamed c
    // twice — depressing the measured ceiling and miscounting traffic.
    // Now a pure single-pass triad sym ← a + 2.5·b over persistent
    // state: exactly 3·8·n² bytes per call, no allocation. (sym is
    // sacrificed as the destination — don't run symmetric-eigen benches
    // on the same instance after this probe.)
    let s = state_mut();
    let n = s.n;
    let acs = s.n;
    let bcs = s.n;
    let ccs = s.n;
    let ap = s.a.as_ptr();
    let bp = s.b.as_ptr();
    let cp = s.sym.as_mut_ptr();
    for j in 0..n {
        unsafe {
            triad(cp.add(j * ccs), ap.add(j * acs), bp.add(j * bcs), n);
        }
    }
    s.sym[0] + s.sym[(n - 1) * n + n - 1]
}

/// c ← a + 2.5·b, one pass, 2 lanes 2× unrolled.
#[cfg_attr(target_arch = "wasm32", target_feature(enable = "simd128"))]
unsafe fn triad(c: *mut f64, a: *const f64, b: *const f64, len: usize) {
    #[cfg(target_arch = "wasm32")]
    {
        use core::arch::wasm32::*;
        let k = f64x2_splat(2.5);
        let mut i = 0usize;
        while i + 4 <= len {
            let a0 = v128_load(a.add(i) as *const v128);
            let b0 = v128_load(b.add(i) as *const v128);
            let a1 = v128_load(a.add(i + 2) as *const v128);
            let b1 = v128_load(b.add(i + 2) as *const v128);
            v128_store(c.add(i) as *mut v128, f64x2_add(a0, f64x2_mul(b0, k)));
            v128_store(c.add(i + 2) as *mut v128, f64x2_add(a1, f64x2_mul(b1, k)));
            i += 4;
        }
        while i < len {
            *c.add(i) = *a.add(i) + *b.add(i) * 2.5;
            i += 1;
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        for i in 0..len {
            *c.add(i) = *a.add(i) + *b.add(i) * 2.5;
        }
    }
}

/// Peak-arithmetic probe: register-resident mul+add chains, 8 independent
/// v128 accumulators × `iters` rounds. FLOPs per call = iters · 8 · 2 lanes
/// · 2 ops. Fused (one relaxed_madd = still 2 FLOPs) when the build has
/// relaxed-simd — so this probe measures the ceiling OF THE BUILD.
#[no_mangle]
pub extern "C" fn run_ceiling_flops(iters: usize) -> f64 {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        ceiling_flops_imp(iters)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = iters;
        f64::NAN
    }
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
// index loops kept verbatim (clippy suggests iterators): this probe is a
// calibrated instrument — its measured ceilings are published — so no
// cosmetic reshaping of its codegen.
#[allow(clippy::needless_range_loop)]
unsafe fn ceiling_flops_imp(iters: usize) -> f64 {
    use core::arch::wasm32::*;
    let m = f64x2_splat(1.000000001);
    let a = f64x2_splat(1e-9);
    let mut acc = [f64x2_splat(1.0); 8];
    for _ in 0..iters {
        for k in 0..8 {
            #[cfg(target_feature = "relaxed-simd")]
            {
                acc[k] = f64x2_relaxed_madd(acc[k], m, a);
            }
            #[cfg(not(target_feature = "relaxed-simd"))]
            {
                acc[k] = f64x2_add(f64x2_mul(acc[k], m), a);
            }
        }
    }
    let mut s = 0.0;
    for k in 0..8 {
        s += f64x2_extract_lane::<0>(acc[k]) + f64x2_extract_lane::<1>(acc[k]);
    }
    s
}



// ============================================================
// f32 BLAS-layer rows and probes — one-for-one twins of the f64
// exports above (same ops, same constants, same probe recipes with
// values cast to f32), reading the *32 state fields. Extern returns
// stay f64 (the wasm ABI the scripts read); the f32 result is cast
// once at the end.
// ============================================================

#[no_mangle]
pub extern "C" fn run_l1_layer_f32(op: usize) -> f64 {
    use blas_wasm::L1 as l1;
    let s = state_mut();
    let n = s.n;
    let acs = s.n;
    let bcs = s.n;
    let ap = s.a32.as_mut_ptr();
    let bp = s.b32.as_mut_ptr();
    let col = |p: *mut f32, cs: usize, j: usize| unsafe {
        core::slice::from_raw_parts_mut(p.add(j * cs), n)
    };
    let mut sink = 0.0f64;
    match op {
        0 => {
            for j in 0..n {
                l1::scopy(col(ap, acs, j), col(bp, bcs, j)).unwrap();
            }
            sink += unsafe { *bp } as f64;
        }
        1 => {
            for j in 0..n {
                l1::sswap(col(ap, acs, j), col(bp, bcs, j)).unwrap();
            }
            sink += unsafe { *ap } as f64;
        }
        2 => {
            for j in 0..n {
                l1::sscal(-1.0, col(ap, acs, j));
            }
            sink += unsafe { *ap } as f64;
        }
        3 => {
            for j in 0..n {
                l1::saxpy(0.001, col(ap, acs, j), col(bp, bcs, j)).unwrap();
            }
            sink += unsafe { *bp } as f64;
        }
        4 => {
            for j in 0..n {
                l1::srot(col(ap, acs, j), col(bp, bcs, j), 0.8, 0.6).unwrap();
            }
            sink += unsafe { *ap } as f64;
        }
        5 => {
            for j in 0..n {
                sink += l1::sdot(col(ap, acs, j), col(bp, bcs, j)).unwrap() as f64;
            }
        }
        6 => {
            for j in 0..n {
                sink += l1::snrm2(col(ap, acs, j)) as f64;
            }
        }
        7 => {
            for j in 0..n {
                sink += l1::sasum(col(ap, acs, j)) as f64;
            }
        }
        8 => {
            for j in 0..n {
                sink += l1::isamax(col(ap, acs, j)) as f64;
            }
        }
        _ => return f64::NAN,
    }
    sink
}

#[no_mangle]
pub extern "C" fn run_l2_layer_f32(op: usize) -> f64 {
    use blas_wasm::L1 as l1;
    use blas_wasm::L2 as l2;
    let s = state_mut();
    let n = s.n;
    let acs = s.n;
    let scs = s.n;
    let tcs = s.n;
    let a_len = if n == 0 { 0 } else { acs * (n - 1) + n };
    let s_len = if n == 0 { 0 } else { scs * (n - 1) + n };
    let t_len = if n == 0 { 0 } else { tcs * (n - 1) + n };
    let a = unsafe { core::slice::from_raw_parts(s.a32.as_ptr(), a_len) };
    let bx = unsafe { core::slice::from_raw_parts(s.b32.as_ptr(), n) };
    let sym = unsafe { core::slice::from_raw_parts_mut(s.sym32.as_mut_ptr(), s_len) };
    let tri = unsafe { core::slice::from_raw_parts(s.tri32.as_ptr(), t_len) };
    let y = unsafe { core::slice::from_raw_parts_mut(s.rhs32.as_mut_ptr(), n) };
    match op {
        0 => l2::sgemv(Trans::NoTrans, n, n, 0.001, a, acs, bx, 0.5, y).unwrap(),
        1 => l2::sgemv(Trans::Trans, n, n, 0.001, a, acs, bx, 0.5, y).unwrap(),
        2 => l2::sger(n, n, 0.0001, bx, bx, sym, scs).unwrap(),
        3 => l2::ssymv(Uplo::Upper, n, 0.001, sym, scs, bx, 0.5, y).unwrap(),
        4 => {
            l1::scopy(bx, y).unwrap();
            l2::strmv(Uplo::Lower, Trans::NoTrans, Diag::NonUnit, n, tri, tcs, y).unwrap();
        }
        5 => {
            l1::scopy(bx, y).unwrap();
            l2::strsv(Uplo::Lower, Trans::NoTrans, Diag::NonUnit, n, tri, tcs, y).unwrap();
        }
        6 => l2::ssyr(Uplo::Upper, n, 0.0001, bx, sym, scs).unwrap(),
        7 => l2::ssyr2(Uplo::Upper, n, 0.0001, bx, bx, sym, scs).unwrap(),
        _ => return f64::NAN,
    }
    (y[0] + sym[0] + y[n - 1]) as f64
}

#[no_mangle]
pub extern "C" fn run_l3_layer_f32(op: usize) -> f64 {
    use blas_wasm::L1 as l1;
    use blas_wasm::L3 as l3;
    let s = state_mut();
    let n = s.n;
    let acs = s.n;
    let bcs = s.n;
    let scs = s.n;
    let tcs = s.n;
    let len = |cs: usize| if n == 0 { 0 } else { cs * (n - 1) + n };
    let a = unsafe { core::slice::from_raw_parts(s.a32.as_ptr(), len(acs)) };
    let b = unsafe { core::slice::from_raw_parts(s.b32.as_ptr(), len(bcs)) };
    let tri = unsafe { core::slice::from_raw_parts(s.tri32.as_ptr(), len(tcs)) };
    let sym = unsafe { core::slice::from_raw_parts_mut(s.sym32.as_mut_ptr(), len(scs)) };
    let refresh = |dst: &mut [f32]| {
        for j in 0..n {
            l1::scopy(&b[j * bcs..j * bcs + n], &mut dst[j * scs..j * scs + n]).unwrap();
        }
    };
    match op {
        0 => l3::sgemm(Trans::NoTrans, Trans::NoTrans, n, n, n, 0.001, a, acs, b, bcs, 0.5, sym, scs, &mut vec![0.0f32; l3::sgemm_work(Trans::NoTrans, Trans::NoTrans, n, n, n)]).unwrap(),
        1 => l3::ssymm(Side::Left, Uplo::Upper, n, n, 0.001, tri, tcs, b, bcs, 0.5, sym, scs).unwrap(),
        2 => l3::ssyrk(Uplo::Upper, Trans::NoTrans, n, n, 0.001, a, acs, 0.5, sym, scs, &mut vec![0.0f32; l3::ssyrk_work(Trans::NoTrans, n, n)]).unwrap(),
        3 => l3::ssyr2k(Uplo::Upper, Trans::NoTrans, n, n, 0.001, a, acs, b, bcs, 0.5, sym, scs, &mut vec![0.0f32; l3::ssyr2k_work(Trans::NoTrans, n, n)]).unwrap(),
        4 => {
            refresh(sym);
            l3::strmm(Side::Left, Uplo::Lower, Trans::NoTrans, Diag::NonUnit, n, n, 1.0, tri, tcs, sym, scs, &mut vec![0.0f32; l3::strmm_work(Side::Left, Trans::NoTrans, n, n)]).unwrap();
        }
        5 => {
            refresh(sym);
            l3::strsm(Side::Left, Uplo::Lower, Trans::NoTrans, Diag::NonUnit, n, n, 1.0, tri, tcs, sym, scs, &mut vec![0.0f32; l3::strsm_work(Side::Left, Trans::NoTrans, n, n)]).unwrap();
        }
        6 => {
            refresh(sym);
            l3::strmm(Side::Right, Uplo::Upper, Trans::NoTrans, Diag::NonUnit, n, n, 1.0, tri, tcs, sym, scs, &mut vec![0.0f32; l3::strmm_work(Side::Right, Trans::NoTrans, n, n)]).unwrap();
        }
        7 => {
            refresh(sym);
            l3::strsm(Side::Right, Uplo::Upper, Trans::NoTrans, Diag::NonUnit, n, n, 1.0, tri, tcs, sym, scs, &mut vec![0.0f32; l3::strsm_work(Side::Right, Trans::NoTrans, n, n)]).unwrap();
        }
        _ => return f64::NAN,
    }
    (sym[0] + sym[len(scs) - 1]) as f64
}

/// f32 L1 determinism probe — same LCG recipe as run_l1_probe with
/// values cast to f32. op: 0 dot, 1 asum, 2 nrm2, 3 iamax.
#[no_mangle]
pub extern "C" fn run_l1_probe_f32(op: usize) -> f64 {
    use blas_wasm::L1 as l1;
    const LEN: usize = 1001;
    let mut x = [0.0f32; LEN];
    let mut y = [0.0f32; LEN];
    let mut s = 42u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32
    };
    for v in x.iter_mut() {
        *v = next();
    }
    for v in y.iter_mut() {
        *v = next();
    }
    match op {
        0 => l1::sdot(&x, &y).unwrap() as f64,
        1 => l1::sasum(&x) as f64,
        2 => l1::snrm2(&x) as f64,
        3 => l1::isamax(&x) as f64,
        _ => f64::NAN,
    }
}

/// f32 L2 determinism probes — same 257×257 recipe cast to f32.
#[no_mangle]
pub extern "C" fn run_l2_probe_f32(op: usize) -> f64 {
    use blas_wasm::L1 as l1;
    use blas_wasm::L2 as l2;
    const N: usize = 257;
    let mut s = 7u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32
    };
    let mut a = alloc_mat_f32(N, &mut next);
    for i in 0..N {
        a[i * N + i] = 2.0 * N as f32 + 1.0;
    }
    let cs = N;
    let a_len = cs * (N - 1) + N;
    let x: [f32; N] = core::array::from_fn(|_| next());
    let mut y: [f32; N] = core::array::from_fn(|_| next());
    let am = &mut a[..a_len];
    match op {
        0 => l2::sgemv(Trans::NoTrans, N, N, 0.7, am, cs, &x, 0.3, &mut y).unwrap(),
        1 => l2::sgemv(Trans::Trans, N, N, 0.7, am, cs, &x, 0.3, &mut y).unwrap(),
        2 => l2::sger(N, N, 0.7, &x, &x, am, cs).unwrap(),
        3 => l2::ssymv(Uplo::Upper, N, 0.7, am, cs, &x, 0.3, &mut y).unwrap(),
        4 => {
            y.copy_from_slice(&x);
            l2::strmv(Uplo::Lower, Trans::NoTrans, Diag::NonUnit, N, am, cs, &mut y).unwrap();
        }
        5 => {
            y.copy_from_slice(&x);
            l2::strsv(Uplo::Lower, Trans::NoTrans, Diag::NonUnit, N, am, cs, &mut y).unwrap();
        }
        6 => l2::ssyr(Uplo::Upper, N, 0.7, &x, am, cs).unwrap(),
        7 => l2::ssyr2(Uplo::Upper, N, 0.7, &x, &x, am, cs).unwrap(),
        8 => {
            y.copy_from_slice(&x);
            l2::strmv(Uplo::Lower, Trans::Trans, Diag::NonUnit, N, am, cs, &mut y).unwrap();
        }
        9 => {
            y.copy_from_slice(&x);
            l2::strsv(Uplo::Lower, Trans::Trans, Diag::NonUnit, N, am, cs, &mut y).unwrap();
        }
        _ => return f64::NAN,
    }
    let a_probe = &a[..N];
    (l1::sasum(&y) + y[0] + y[N - 1] + l1::sasum(a_probe)) as f64
}

/// f32 L3 determinism probes — same 65×65 recipe cast to f32.
#[no_mangle]
pub extern "C" fn run_l3_probe_f32(op: usize) -> f64 {
    use blas_wasm::L1 as l1;
    use blas_wasm::L3 as l3;
    const N: usize = 65;
    const K: usize = 33;
    let mut st = 11u64;
    let mut next = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((st >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32
    };
    let mut a = alloc_mat_f32(N, &mut next);
    for i in 0..N {
        a[i * N + i] = 2.0 * N as f32 + 1.0;
    }
    let b = alloc_mat_f32(N, &mut next);
    let mut c = alloc_mat_f32(N, &mut next);
    let (acs, bcs, ccs) = (N, N, N);
    let len = |cs: usize| cs * (N - 1) + N;
    let av = &a[..len(acs)];
    let bv = &b[..len(bcs)];
    let cv = &mut c[..len(ccs)];
    match op {
        0 => l3::sgemm(Trans::NoTrans, Trans::NoTrans, N, N, K, 0.7, av, acs, bv, bcs, 0.3, cv, ccs, &mut vec![0.0f32; l3::sgemm_work(Trans::NoTrans, Trans::NoTrans, N, N, K)]).unwrap(),
        1 => l3::ssymm(Side::Left, Uplo::Upper, N, N, 0.7, av, acs, bv, bcs, 0.3, cv, ccs).unwrap(),
        2 => l3::ssyrk(Uplo::Upper, Trans::NoTrans, N, K, 0.7, av, acs, 0.3, cv, ccs, &mut vec![0.0f32; l3::ssyrk_work(Trans::NoTrans, N, K)]).unwrap(),
        3 => l3::ssyr2k(Uplo::Upper, Trans::NoTrans, N, K, 0.7, av, acs, bv, bcs, 0.3, cv, ccs, &mut vec![0.0f32; l3::ssyr2k_work(Trans::NoTrans, N, K)]).unwrap(),
        4 => l3::strmm(Side::Left, Uplo::Lower, Trans::NoTrans, Diag::NonUnit, N, N, 0.7, av, acs, cv, ccs, &mut vec![0.0f32; l3::strmm_work(Side::Left, Trans::NoTrans, N, N)]).unwrap(),
        5 => l3::strsm(Side::Left, Uplo::Lower, Trans::NoTrans, Diag::NonUnit, N, N, 0.7, av, acs, cv, ccs, &mut vec![0.0f32; l3::strsm_work(Side::Left, Trans::NoTrans, N, N)]).unwrap(),
        6 => l3::strmm(Side::Right, Uplo::Upper, Trans::NoTrans, Diag::NonUnit, N, N, 0.7, av, acs, cv, ccs, &mut vec![0.0f32; l3::strmm_work(Side::Right, Trans::NoTrans, N, N)]).unwrap(),
        7 => l3::strsm(Side::Right, Uplo::Upper, Trans::NoTrans, Diag::NonUnit, N, N, 0.7, av, acs, cv, ccs, &mut vec![0.0f32; l3::strsm_work(Side::Right, Trans::NoTrans, N, N)]).unwrap(),
        8 => l3::ssymm(Side::Right, Uplo::Upper, N, N, 0.7, av, acs, bv, bcs, 0.3, cv, ccs).unwrap(),
        // the transposed forms (todos S1): the copy zone, the packed
        // transposed read (a shape above this type's packed threshold,
        // its own fold), the syrk family, trmm and trsm on both sides
        9 => l3::sgemm(Trans::Trans, Trans::NoTrans, N, N, K, 0.7, av, acs, bv, bcs, 0.3, cv, ccs, &mut vec![0.0f32; l3::sgemm_work(Trans::Trans, Trans::NoTrans, N, N, K)]).unwrap(),
        10 => {
            const PK: usize = 900;
            let ab = alloc_mat_f32(PK, &mut next);
            let bb = alloc_mat_f32(PK, &mut next);
            let mut cb = alloc_mat_f32(PK, &mut next);
            let cb = &mut cb[..PK * 8];
            l3::sgemm(Trans::Trans, Trans::NoTrans, PK, 8, PK, 0.7, &ab, PK, &bb[..PK * 8], PK, 0.3, cb, PK, &mut vec![0.0f32; l3::sgemm_work(Trans::Trans, Trans::NoTrans, PK, 8, PK)]).unwrap();
            let mut fold = 0.0f64;
            for j in 0..8 {
                fold += l1::sasum(&cb[j * PK..j * PK + PK]) as f64;
            }
            return fold;
        }
        11 => l3::ssyrk(Uplo::Upper, Trans::Trans, N, K, 0.7, av, acs, 0.3, cv, ccs, &mut vec![0.0f32; l3::ssyrk_work(Trans::Trans, N, K)]).unwrap(),
        12 => l3::strmm(Side::Left, Uplo::Lower, Trans::Trans, Diag::NonUnit, N, N, 0.7, av, acs, cv, ccs, &mut vec![0.0f32; l3::strmm_work(Side::Left, Trans::Trans, N, N)]).unwrap(),
        13 => l3::strsm(Side::Left, Uplo::Lower, Trans::Trans, Diag::NonUnit, N, N, 0.7, av, acs, cv, ccs, &mut vec![0.0f32; l3::strsm_work(Side::Left, Trans::Trans, N, N)]).unwrap(),
        14 => l3::strmm(Side::Right, Uplo::Upper, Trans::Trans, Diag::NonUnit, N, N, 0.7, av, acs, cv, ccs, &mut vec![0.0f32; l3::strmm_work(Side::Right, Trans::Trans, N, N)]).unwrap(),
        15 => l3::strsm(Side::Right, Uplo::Upper, Trans::Trans, Diag::NonUnit, N, N, 0.7, av, acs, cv, ccs, &mut vec![0.0f32; l3::strsm_work(Side::Right, Trans::Trans, N, N)]).unwrap(),
        _ => return f64::NAN,
    }
    let mut fold = 0.0f32;
    for j in 0..N {
        fold += l1::sasum(&cv[j * ccs..j * ccs + N]);
    }
    (fold + cv[0] + cv[len(ccs) - 1]) as f64
}

/// f32 peak-arithmetic probe: f32x4 twin of run_ceiling_flops.
/// FLOPs per call = iters · 8 · 4 lanes · 2 ops.
#[no_mangle]
pub extern "C" fn run_ceiling_flops_f32(iters: usize) -> f64 {
    #[cfg(target_arch = "wasm32")]
    unsafe {
        ceiling_flops_f32_imp(iters)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        let _ = iters;
        f64::NAN
    }
}

#[cfg(target_arch = "wasm32")]
#[target_feature(enable = "simd128")]
// same calibrated-instrument rule as ceiling_flops_imp
#[allow(clippy::needless_range_loop)]
unsafe fn ceiling_flops_f32_imp(iters: usize) -> f64 {
    use core::arch::wasm32::*;
    let m = f32x4_splat(1.0000001);
    let a = f32x4_splat(1e-7);
    let mut acc = [f32x4_splat(1.0); 8];
    for _ in 0..iters {
        for k in 0..8 {
            #[cfg(target_feature = "relaxed-simd")]
            {
                acc[k] = f32x4_relaxed_madd(acc[k], m, a);
            }
            #[cfg(not(target_feature = "relaxed-simd"))]
            {
                acc[k] = f32x4_add(f32x4_mul(acc[k], m), a);
            }
        }
    }
    let mut s = 0.0f32;
    for k in 0..8 {
        s += f32x4_extract_lane::<0>(acc[k])
            + f32x4_extract_lane::<1>(acc[k])
            + f32x4_extract_lane::<2>(acc[k])
            + f32x4_extract_lane::<3>(acc[k]);
    }
    s as f64
}

// ============================================================
// c64 BLAS-layer rows and probes — twins of the f64 exports above
// over the *z state fields (16-byte elements; complex constants pick
// the same magnitudes). Extern returns stay f64; complex results are
// folded re+im at the end.
// ============================================================

#[no_mangle]
pub extern "C" fn run_l1_layer_z(op: usize) -> f64 {
    use blas_wasm::L1 as l1;
    let s = state_mut();
    let n = s.n;
    let ap = s.az.as_mut_ptr();
    let bp = s.bz.as_mut_ptr();
    let col = |p: *mut C64, j: usize| unsafe { core::slice::from_raw_parts_mut(p.add(j * n), n) };
    let mut sink = 0.0f64;
    match op {
        0 => {
            for j in 0..n {
                l1::zcopy(col(ap, j), col(bp, j)).unwrap();
            }
            sink += unsafe { (*bp).re };
        }
        1 => {
            for j in 0..n {
                l1::zswap(col(ap, j), col(bp, j)).unwrap();
            }
            sink += unsafe { (*ap).re };
        }
        2 => {
            for j in 0..n {
                l1::zscal(C64::new(-1.0, 0.0), col(ap, j));
            }
            sink += unsafe { (*ap).re };
        }
        3 => {
            for j in 0..n {
                l1::zdscal(-1.0, col(ap, j));
            }
            sink += unsafe { (*ap).re };
        }
        4 => {
            for j in 0..n {
                l1::zaxpy(C64::new(0.001, 0.0), col(ap, j), col(bp, j)).unwrap();
            }
            sink += unsafe { (*bp).re };
        }
        5 => {
            for j in 0..n {
                l1::zdrot(col(ap, j), col(bp, j), 0.8, 0.6).unwrap();
            }
            sink += unsafe { (*ap).re };
        }
        6 => {
            for j in 0..n {
                let d = l1::zdotu(col(ap, j), col(bp, j)).unwrap();
                sink += d.re + d.im;
            }
        }
        7 => {
            for j in 0..n {
                let d = l1::zdotc(col(ap, j), col(bp, j)).unwrap();
                sink += d.re + d.im;
            }
        }
        8 => {
            for j in 0..n {
                sink += l1::dznrm2(col(ap, j));
            }
        }
        9 => {
            for j in 0..n {
                sink += l1::dzasum(col(ap, j));
            }
        }
        10 => {
            for j in 0..n {
                sink += l1::izamax(col(ap, j)) as f64;
            }
        }
        _ => return f64::NAN,
    }
    sink
}

/// Level-2 c64 roofline rows. op: 0 gemv, 1 gemv_t, 2 gemv_c, 3 geru,
/// 4 gerc, 5 hemv, 6 trmv, 7 trsv, 8 her, 9 her2. Same
/// bounded-value discipline as the f64 rows (small alpha on
/// accumulating updates; trmv/trsv re-copy x fresh each call).
#[no_mangle]
pub extern "C" fn run_l2_layer_z(op: usize) -> f64 {
    use blas_wasm::L1 as l1;
    use blas_wasm::L2 as l2;
    let s = state_mut();
    let n = s.n;
    let cs = s.n;
    let len = if n == 0 { 0 } else { cs * (n - 1) + n };
    let a = unsafe { core::slice::from_raw_parts(s.az.as_ptr(), len) };
    let bx = unsafe { core::slice::from_raw_parts(s.bz.as_ptr(), n) };
    let sym = unsafe { core::slice::from_raw_parts_mut(s.symz.as_mut_ptr(), len) };
    let tri = unsafe { core::slice::from_raw_parts(s.triz.as_ptr(), len) };
    let y = unsafe { core::slice::from_raw_parts_mut(s.rhsz.as_mut_ptr(), n) };
    let al = C64::new(0.001, 0.0);
    let al4 = C64::new(0.0001, 0.0);
    let be = C64::new(0.5, 0.0);
    match op {
        0 => l2::zgemv(Trans::NoTrans, n, n, al, a, cs, bx, be, y).unwrap(),
        1 => l2::zgemv(Trans::Trans, n, n, al, a, cs, bx, be, y).unwrap(),
        2 => l2::zgemv(Trans::ConjTrans, n, n, al, a, cs, bx, be, y).unwrap(),
        3 => l2::zgeru(n, n, al4, bx, bx, sym, cs).unwrap(),
        4 => l2::zgerc(n, n, al4, bx, bx, sym, cs).unwrap(),
        5 => l2::zhemv(Uplo::Upper, n, al, sym, cs, bx, be, y).unwrap(),
        6 => {
            l1::zcopy(bx, y).unwrap();
            l2::ztrmv(Uplo::Lower, Trans::NoTrans, Diag::NonUnit, n, tri, cs, y).unwrap();
        }
        7 => {
            l1::zcopy(bx, y).unwrap();
            l2::ztrsv(Uplo::Lower, Trans::NoTrans, Diag::NonUnit, n, tri, cs, y).unwrap();
        }
        8 => l2::zher(Uplo::Upper, n, 0.0001, bx, sym, cs).unwrap(),
        9 => l2::zher2(Uplo::Upper, n, al4, bx, bx, sym, cs).unwrap(),
        _ => return f64::NAN,
    }
    y[0].re + y[0].im + sym[0].re + y[n - 1].re
}

/// Level-3 c64 roofline rows. op: 0 gemm, 1 hemm_left, 2 herk,
/// 3 her2k, 4 trmm_left, 5 trsm_left, 6 trmm_right, 7 trsm_right.
#[no_mangle]
pub extern "C" fn run_l3_layer_z(op: usize) -> f64 {
    use blas_wasm::L1 as l1;
    use blas_wasm::L3 as l3;
    let s = state_mut();
    let n = s.n;
    let cs = s.n;
    let len = if n == 0 { 0 } else { cs * (n - 1) + n };
    let a = unsafe { core::slice::from_raw_parts(s.az.as_ptr(), len) };
    let b = unsafe { core::slice::from_raw_parts(s.bz.as_ptr(), len) };
    let tri = unsafe { core::slice::from_raw_parts(s.triz.as_ptr(), len) };
    let sym = unsafe { core::slice::from_raw_parts_mut(s.symz.as_mut_ptr(), len) };
    let refresh = |dst: &mut [C64]| {
        for j in 0..n {
            l1::zcopy(&b[j * cs..j * cs + n], &mut dst[j * cs..j * cs + n]).unwrap();
        }
    };
    let al = C64::new(0.001, 0.0);
    let be = C64::new(0.5, 0.0);
    let one = C64::ONE;
    match op {
        0 => l3::zgemm(Trans::NoTrans, Trans::NoTrans, n, n, n, al, a, cs, b, cs, be, sym, cs, &mut vec![C64::ZERO; l3::zgemm_work(Trans::NoTrans, Trans::NoTrans, n, n, n)]).unwrap(),
        1 => l3::zhemm(Side::Left, Uplo::Upper, n, n, al, tri, cs, b, cs, be, sym, cs).unwrap(),
        2 => l3::zherk(Uplo::Upper, Trans::NoTrans, n, n, 0.001, a, cs, 0.5, sym, cs, &mut vec![C64::ZERO; l3::zherk_work(Trans::NoTrans, n, n)]).unwrap(),
        3 => l3::zher2k(Uplo::Upper, Trans::NoTrans, n, n, al, a, cs, b, cs, 0.5, sym, cs, &mut vec![C64::ZERO; l3::zher2k_work(Trans::NoTrans, n, n)]).unwrap(),
        4 => {
            refresh(sym);
            l3::ztrmm(Side::Left, Uplo::Lower, Trans::NoTrans, Diag::NonUnit, n, n, one, tri, cs, sym, cs, &mut vec![C64::ZERO; l3::ztrmm_work(Side::Left, Trans::NoTrans, n, n)]).unwrap();
        }
        5 => {
            refresh(sym);
            l3::ztrsm(Side::Left, Uplo::Lower, Trans::NoTrans, Diag::NonUnit, n, n, one, tri, cs, sym, cs, &mut vec![C64::ZERO; l3::ztrsm_work(Side::Left, Trans::NoTrans, n, n)]).unwrap();
        }
        6 => {
            refresh(sym);
            l3::ztrmm(Side::Right, Uplo::Upper, Trans::NoTrans, Diag::NonUnit, n, n, one, tri, cs, sym, cs, &mut vec![C64::ZERO; l3::ztrmm_work(Side::Right, Trans::NoTrans, n, n)]).unwrap();
        }
        7 => {
            refresh(sym);
            l3::ztrsm(Side::Right, Uplo::Upper, Trans::NoTrans, Diag::NonUnit, n, n, one, tri, cs, sym, cs, &mut vec![C64::ZERO; l3::ztrsm_work(Side::Right, Trans::NoTrans, n, n)]).unwrap();
        }
        _ => return f64::NAN,
    }
    sym[0].re + sym[0].im + sym[len - 1].re
}

/// c64 cross-target determinism probes, L1: fixed LCG data (len 1001 —
/// odd, so the scalar tails run), one reduction per op. op: 0 dotu,
/// 1 dotc, 2 nrm2, 3 asum, 4 iamax.
#[no_mangle]
pub extern "C" fn run_l1_probe_z(op: usize) -> f64 {
    use blas_wasm::L1 as l1;
    const LEN: usize = 1001;
    let mut s = 43u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut mk = || -> Vec<C64> {
        (0..LEN)
            .map(|_| {
                let re = next();
                let im = next();
                C64::new(re, im)
            })
            .collect()
    };
    let x = mk();
    let y = mk();
    match op {
        0 => {
            let d = l1::zdotu(&x, &y).unwrap();
            d.re + d.im
        }
        1 => {
            let d = l1::zdotc(&x, &y).unwrap();
            d.re + d.im
        }
        2 => l1::dznrm2(&x),
        3 => l1::dzasum(&x),
        4 => l1::izamax(&x) as f64,
        _ => f64::NAN,
    }
}

/// c64 L2 determinism probes: fixed LCG-filled 257×257 complex matrix,
/// one op each, folded with the layer's own dzasum. op: 0 gemv,
/// 1 gemv_t, 2 gemv_c, 3 geru, 4 gerc, 5 hemv, 6 trmv, 7 trsv,
/// 8 her, 9 her2.
#[no_mangle]
pub extern "C" fn run_l2_probe_z(op: usize) -> f64 {
    use blas_wasm::L1 as l1;
    use blas_wasm::L2 as l2;
    const N: usize = 257;
    let mut s = 8u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut a = alloc_mat_c64(N, &mut next);
    for i in 0..N {
        a[i * N + i] = C64::new(2.0 * N as f64 + 1.0, 0.7); // solve-safe diagonal
    }
    let cs = N;
    let a_len = cs * (N - 1) + N;
    let x: Vec<C64> = (0..N)
        .map(|_| {
            let re = next();
            let im = next();
            C64::new(re, im)
        })
        .collect();
    let mut y: Vec<C64> = (0..N)
        .map(|_| {
            let re = next();
            let im = next();
            C64::new(re, im)
        })
        .collect();
    let am = &mut a[..a_len];
    let al = C64::new(0.7, -0.2);
    let be = C64::new(0.3, 0.1);
    match op {
        0 => l2::zgemv(Trans::NoTrans, N, N, al, am, cs, &x, be, &mut y).unwrap(),
        1 => l2::zgemv(Trans::Trans, N, N, al, am, cs, &x, be, &mut y).unwrap(),
        2 => l2::zgemv(Trans::ConjTrans, N, N, al, am, cs, &x, be, &mut y).unwrap(),
        3 => l2::zgeru(N, N, al, &x, &x, am, cs).unwrap(),
        4 => l2::zgerc(N, N, al, &x, &x, am, cs).unwrap(),
        5 => l2::zhemv(Uplo::Upper, N, al, am, cs, &x, be, &mut y).unwrap(),
        6 => {
            l1::zcopy(&x, &mut y).unwrap();
            l2::ztrmv(Uplo::Lower, Trans::NoTrans, Diag::NonUnit, N, am, cs, &mut y).unwrap();
        }
        7 => {
            l1::zcopy(&x, &mut y).unwrap();
            l2::ztrsv(Uplo::Lower, Trans::NoTrans, Diag::NonUnit, N, am, cs, &mut y).unwrap();
        }
        8 => l2::zher(Uplo::Upper, N, 0.7, &x, am, cs).unwrap(),
        9 => l2::zher2(Uplo::Upper, N, al, &x, &x, am, cs).unwrap(),
        10 => {
            l1::zcopy(&x, &mut y).unwrap();
            l2::ztrmv(Uplo::Lower, Trans::ConjTrans, Diag::NonUnit, N, am, cs, &mut y).unwrap();
        }
        11 => {
            l1::zcopy(&x, &mut y).unwrap();
            l2::ztrsv(Uplo::Lower, Trans::ConjTrans, Diag::NonUnit, N, am, cs, &mut y).unwrap();
        }
        _ => return f64::NAN,
    }
    let a_probe = &a[..N];
    l1::dzasum(&y) + y[0].re + y[0].im + y[N - 1].re + l1::dzasum(a_probe)
}

/// c64 L3 determinism probes: fixed LCG-filled 65×65 complex matrices
/// (odd — tails everywhere), one op each. op order matches
/// run_l3_layer_z plus hemm_right at 8.
#[no_mangle]
pub extern "C" fn run_l3_probe_z(op: usize) -> f64 {
    use blas_wasm::L1 as l1;
    use blas_wasm::L3 as l3;
    const N: usize = 65;
    const K: usize = 33;
    let mut st = 12u64;
    let mut next = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((st >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0
    };
    let mut a = alloc_mat_c64(N, &mut next);
    for i in 0..N {
        a[i * N + i] = C64::new(2.0 * N as f64 + 1.0, 0.7);
    }
    let b = alloc_mat_c64(N, &mut next);
    let mut c = alloc_mat_c64(N, &mut next);
    let cs = N;
    let len = cs * (N - 1) + N;
    let av = &a[..len];
    let bv = &b[..len];
    let cv = &mut c[..len];
    let al = C64::new(0.7, -0.2);
    let be = C64::new(0.3, 0.1);
    match op {
        0 => l3::zgemm(Trans::NoTrans, Trans::NoTrans, N, N, K, al, av, cs, bv, cs, be, cv, cs, &mut vec![C64::ZERO; l3::zgemm_work(Trans::NoTrans, Trans::NoTrans, N, N, K)]).unwrap(),
        1 => l3::zhemm(Side::Left, Uplo::Upper, N, N, al, av, cs, bv, cs, be, cv, cs).unwrap(),
        2 => l3::zherk(Uplo::Upper, Trans::NoTrans, N, K, 0.7, av, cs, 0.3, cv, cs, &mut vec![C64::ZERO; l3::zherk_work(Trans::NoTrans, N, K)]).unwrap(),
        3 => l3::zher2k(Uplo::Upper, Trans::NoTrans, N, K, al, av, cs, bv, cs, 0.3, cv, cs, &mut vec![C64::ZERO; l3::zher2k_work(Trans::NoTrans, N, K)]).unwrap(),
        4 => l3::ztrmm(Side::Left, Uplo::Lower, Trans::NoTrans, Diag::NonUnit, N, N, al, av, cs, cv, cs, &mut vec![C64::ZERO; l3::ztrmm_work(Side::Left, Trans::NoTrans, N, N)]).unwrap(),
        5 => l3::ztrsm(Side::Left, Uplo::Lower, Trans::NoTrans, Diag::NonUnit, N, N, al, av, cs, cv, cs, &mut vec![C64::ZERO; l3::ztrsm_work(Side::Left, Trans::NoTrans, N, N)]).unwrap(),
        6 => l3::ztrmm(Side::Right, Uplo::Upper, Trans::NoTrans, Diag::NonUnit, N, N, al, av, cs, cv, cs, &mut vec![C64::ZERO; l3::ztrmm_work(Side::Right, Trans::NoTrans, N, N)]).unwrap(),
        7 => l3::ztrsm(Side::Right, Uplo::Upper, Trans::NoTrans, Diag::NonUnit, N, N, al, av, cs, cv, cs, &mut vec![C64::ZERO; l3::ztrsm_work(Side::Right, Trans::NoTrans, N, N)]).unwrap(),
        8 => l3::zhemm(Side::Right, Uplo::Upper, N, N, al, av, cs, bv, cs, be, cv, cs).unwrap(),
        // the transposed forms (todos S1): the copy zone, the packed
        // transposed read (a shape above this type's packed threshold,
        // its own fold), the syrk family, trmm and trsm on both sides
        9 => l3::zgemm(Trans::ConjTrans, Trans::NoTrans, N, N, K, al, av, cs, bv, cs, be, cv, cs, &mut vec![C64::ZERO; l3::zgemm_work(Trans::ConjTrans, Trans::NoTrans, N, N, K)]).unwrap(),
        10 => {
            const PK: usize = 300;
            let ab = alloc_mat_c64(PK, &mut next);
            let bb = alloc_mat_c64(PK, &mut next);
            let mut cb = alloc_mat_c64(PK, &mut next);
            let cb = &mut cb[..PK * 8];
            l3::zgemm(Trans::ConjTrans, Trans::NoTrans, PK, 8, PK, al, &ab, PK, &bb[..PK * 8], PK, be, cb, PK, &mut vec![C64::ZERO; l3::zgemm_work(Trans::ConjTrans, Trans::NoTrans, PK, 8, PK)]).unwrap();
            let mut fold = 0.0f64;
            for j in 0..8 {
                fold += l1::dzasum(&cb[j * PK..j * PK + PK]);
            }
            return fold;
        }
        11 => l3::zherk(Uplo::Upper, Trans::ConjTrans, N, K, 0.7, av, cs, 0.3, cv, cs, &mut vec![C64::ZERO; l3::zherk_work(Trans::ConjTrans, N, K)]).unwrap(),
        12 => l3::ztrmm(Side::Left, Uplo::Lower, Trans::ConjTrans, Diag::NonUnit, N, N, al, av, cs, cv, cs, &mut vec![C64::ZERO; l3::ztrmm_work(Side::Left, Trans::ConjTrans, N, N)]).unwrap(),
        13 => l3::ztrsm(Side::Left, Uplo::Lower, Trans::ConjTrans, Diag::NonUnit, N, N, al, av, cs, cv, cs, &mut vec![C64::ZERO; l3::ztrsm_work(Side::Left, Trans::ConjTrans, N, N)]).unwrap(),
        14 => l3::ztrmm(Side::Right, Uplo::Upper, Trans::ConjTrans, Diag::NonUnit, N, N, al, av, cs, cv, cs, &mut vec![C64::ZERO; l3::ztrmm_work(Side::Right, Trans::ConjTrans, N, N)]).unwrap(),
        15 => l3::ztrsm(Side::Right, Uplo::Upper, Trans::ConjTrans, Diag::NonUnit, N, N, al, av, cs, cv, cs, &mut vec![C64::ZERO; l3::ztrsm_work(Side::Right, Trans::ConjTrans, N, N)]).unwrap(),
        _ => return f64::NAN,
    }
    let mut fold = 0.0;
    for j in 0..N {
        fold += l1::dzasum(&cv[j * cs..j * cs + N]);
    }
    fold + cv[0].re + cv[0].im + cv[len - 1].re
}

// ============================================================
// c32 BLAS-layer rows and probes — twins of the c64 exports above
// over the *c state fields (8-byte elements, two complexes per
// register). Extern returns stay f64; f32 results are widened once
// at the end.
// ============================================================

use blas_wasm::C32;

#[no_mangle]
pub extern "C" fn run_l1_layer_c(op: usize) -> f64 {
    use blas_wasm::L1 as l1;
    let s = state_mut();
    let n = s.n;
    let ap = s.ac.as_mut_ptr();
    let bp = s.bc.as_mut_ptr();
    let col = |p: *mut C32, j: usize| unsafe { core::slice::from_raw_parts_mut(p.add(j * n), n) };
    let mut sink = 0.0f64;
    match op {
        0 => {
            for j in 0..n {
                l1::ccopy(col(ap, j), col(bp, j)).unwrap();
            }
            sink += unsafe { (*bp).re } as f64;
        }
        1 => {
            for j in 0..n {
                l1::cswap(col(ap, j), col(bp, j)).unwrap();
            }
            sink += unsafe { (*ap).re } as f64;
        }
        2 => {
            for j in 0..n {
                l1::cscal(C32::new(-1.0, 0.0), col(ap, j));
            }
            sink += unsafe { (*ap).re } as f64;
        }
        3 => {
            for j in 0..n {
                l1::csscal(-1.0, col(ap, j));
            }
            sink += unsafe { (*ap).re } as f64;
        }
        4 => {
            for j in 0..n {
                l1::caxpy(C32::new(0.001, 0.0), col(ap, j), col(bp, j)).unwrap();
            }
            sink += unsafe { (*bp).re } as f64;
        }
        5 => {
            for j in 0..n {
                l1::csrot(col(ap, j), col(bp, j), 0.8, 0.6).unwrap();
            }
            sink += unsafe { (*ap).re } as f64;
        }
        6 => {
            for j in 0..n {
                let d = l1::cdotu(col(ap, j), col(bp, j)).unwrap();
                sink += (d.re + d.im) as f64;
            }
        }
        7 => {
            for j in 0..n {
                let d = l1::cdotc(col(ap, j), col(bp, j)).unwrap();
                sink += (d.re + d.im) as f64;
            }
        }
        8 => {
            for j in 0..n {
                sink += l1::scnrm2(col(ap, j)) as f64;
            }
        }
        9 => {
            for j in 0..n {
                sink += l1::scasum(col(ap, j)) as f64;
            }
        }
        10 => {
            for j in 0..n {
                sink += l1::icamax(col(ap, j)) as f64;
            }
        }
        _ => return f64::NAN,
    }
    sink
}

/// Level-2 c32 roofline rows — same op order as run_l2_layer_z.
#[no_mangle]
pub extern "C" fn run_l2_layer_c(op: usize) -> f64 {
    use blas_wasm::L1 as l1;
    use blas_wasm::L2 as l2;
    let s = state_mut();
    let n = s.n;
    let cs = s.n;
    let len = if n == 0 { 0 } else { cs * (n - 1) + n };
    let a = unsafe { core::slice::from_raw_parts(s.ac.as_ptr(), len) };
    let bx = unsafe { core::slice::from_raw_parts(s.bc.as_ptr(), n) };
    let sym = unsafe { core::slice::from_raw_parts_mut(s.symc.as_mut_ptr(), len) };
    let tri = unsafe { core::slice::from_raw_parts(s.tric.as_ptr(), len) };
    let y = unsafe { core::slice::from_raw_parts_mut(s.rhsc.as_mut_ptr(), n) };
    let al = C32::new(0.001, 0.0);
    let al4 = C32::new(0.0001, 0.0);
    let be = C32::new(0.5, 0.0);
    match op {
        0 => l2::cgemv(Trans::NoTrans, n, n, al, a, cs, bx, be, y).unwrap(),
        1 => l2::cgemv(Trans::Trans, n, n, al, a, cs, bx, be, y).unwrap(),
        2 => l2::cgemv(Trans::ConjTrans, n, n, al, a, cs, bx, be, y).unwrap(),
        3 => l2::cgeru(n, n, al4, bx, bx, sym, cs).unwrap(),
        4 => l2::cgerc(n, n, al4, bx, bx, sym, cs).unwrap(),
        5 => l2::chemv(Uplo::Upper, n, al, sym, cs, bx, be, y).unwrap(),
        6 => {
            l1::ccopy(bx, y).unwrap();
            l2::ctrmv(Uplo::Lower, Trans::NoTrans, Diag::NonUnit, n, tri, cs, y).unwrap();
        }
        7 => {
            l1::ccopy(bx, y).unwrap();
            l2::ctrsv(Uplo::Lower, Trans::NoTrans, Diag::NonUnit, n, tri, cs, y).unwrap();
        }
        8 => l2::cher(Uplo::Upper, n, 0.0001, bx, sym, cs).unwrap(),
        9 => l2::cher2(Uplo::Upper, n, al4, bx, bx, sym, cs).unwrap(),
        _ => return f64::NAN,
    }
    (y[0].re + y[0].im + sym[0].re + y[n - 1].re) as f64
}

/// Level-3 c32 roofline rows — same op order as run_l3_layer_z.
#[no_mangle]
pub extern "C" fn run_l3_layer_c(op: usize) -> f64 {
    use blas_wasm::L1 as l1;
    use blas_wasm::L3 as l3;
    let s = state_mut();
    let n = s.n;
    let cs = s.n;
    let len = if n == 0 { 0 } else { cs * (n - 1) + n };
    let a = unsafe { core::slice::from_raw_parts(s.ac.as_ptr(), len) };
    let b = unsafe { core::slice::from_raw_parts(s.bc.as_ptr(), len) };
    let tri = unsafe { core::slice::from_raw_parts(s.tric.as_ptr(), len) };
    let sym = unsafe { core::slice::from_raw_parts_mut(s.symc.as_mut_ptr(), len) };
    let refresh = |dst: &mut [C32]| {
        for j in 0..n {
            l1::ccopy(&b[j * cs..j * cs + n], &mut dst[j * cs..j * cs + n]).unwrap();
        }
    };
    let al = C32::new(0.001, 0.0);
    let be = C32::new(0.5, 0.0);
    let one = C32::ONE;
    match op {
        0 => l3::cgemm(Trans::NoTrans, Trans::NoTrans, n, n, n, al, a, cs, b, cs, be, sym, cs, &mut vec![C32::ZERO; l3::cgemm_work(Trans::NoTrans, Trans::NoTrans, n, n, n)]).unwrap(),
        1 => l3::chemm(Side::Left, Uplo::Upper, n, n, al, tri, cs, b, cs, be, sym, cs).unwrap(),
        2 => l3::cherk(Uplo::Upper, Trans::NoTrans, n, n, 0.001, a, cs, 0.5, sym, cs, &mut vec![C32::ZERO; l3::cherk_work(Trans::NoTrans, n, n)]).unwrap(),
        3 => l3::cher2k(Uplo::Upper, Trans::NoTrans, n, n, al, a, cs, b, cs, 0.5, sym, cs, &mut vec![C32::ZERO; l3::cher2k_work(Trans::NoTrans, n, n)]).unwrap(),
        4 => {
            refresh(sym);
            l3::ctrmm(Side::Left, Uplo::Lower, Trans::NoTrans, Diag::NonUnit, n, n, one, tri, cs, sym, cs, &mut vec![C32::ZERO; l3::ctrmm_work(Side::Left, Trans::NoTrans, n, n)]).unwrap();
        }
        5 => {
            refresh(sym);
            l3::ctrsm(Side::Left, Uplo::Lower, Trans::NoTrans, Diag::NonUnit, n, n, one, tri, cs, sym, cs, &mut vec![C32::ZERO; l3::ctrsm_work(Side::Left, Trans::NoTrans, n, n)]).unwrap();
        }
        6 => {
            refresh(sym);
            l3::ctrmm(Side::Right, Uplo::Upper, Trans::NoTrans, Diag::NonUnit, n, n, one, tri, cs, sym, cs, &mut vec![C32::ZERO; l3::ctrmm_work(Side::Right, Trans::NoTrans, n, n)]).unwrap();
        }
        7 => {
            refresh(sym);
            l3::ctrsm(Side::Right, Uplo::Upper, Trans::NoTrans, Diag::NonUnit, n, n, one, tri, cs, sym, cs, &mut vec![C32::ZERO; l3::ctrsm_work(Side::Right, Trans::NoTrans, n, n)]).unwrap();
        }
        _ => return f64::NAN,
    }
    (sym[0].re + sym[0].im + sym[len - 1].re) as f64
}

/// c32 L1 determinism probes — same op order as run_l1_probe_z.
#[no_mangle]
pub extern "C" fn run_l1_probe_c(op: usize) -> f64 {
    use blas_wasm::L1 as l1;
    const LEN: usize = 1001;
    let mut s = 44u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32
    };
    let mut mk = || -> Vec<C32> {
        (0..LEN)
            .map(|_| {
                let re = next();
                let im = next();
                C32::new(re, im)
            })
            .collect()
    };
    let x = mk();
    let y = mk();
    match op {
        0 => {
            let d = l1::cdotu(&x, &y).unwrap();
            (d.re + d.im) as f64
        }
        1 => {
            let d = l1::cdotc(&x, &y).unwrap();
            (d.re + d.im) as f64
        }
        2 => l1::scnrm2(&x) as f64,
        3 => l1::scasum(&x) as f64,
        4 => l1::icamax(&x) as f64,
        _ => f64::NAN,
    }
}

/// c32 L2 determinism probes — same op order as run_l2_probe_z.
#[no_mangle]
pub extern "C" fn run_l2_probe_c(op: usize) -> f64 {
    use blas_wasm::L1 as l1;
    use blas_wasm::L2 as l2;
    const N: usize = 257;
    let mut s = 9u64;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((s >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32
    };
    let mut a = alloc_mat_c32(N, &mut next);
    for i in 0..N {
        a[i * N + i] = C32::new(2.0 * N as f32 + 1.0, 0.7); // solve-safe diagonal
    }
    let cs = N;
    let a_len = cs * (N - 1) + N;
    let x: Vec<C32> = (0..N)
        .map(|_| {
            let re = next();
            let im = next();
            C32::new(re, im)
        })
        .collect();
    let mut y: Vec<C32> = (0..N)
        .map(|_| {
            let re = next();
            let im = next();
            C32::new(re, im)
        })
        .collect();
    let am = &mut a[..a_len];
    let al = C32::new(0.7, -0.2);
    let be = C32::new(0.3, 0.1);
    match op {
        0 => l2::cgemv(Trans::NoTrans, N, N, al, am, cs, &x, be, &mut y).unwrap(),
        1 => l2::cgemv(Trans::Trans, N, N, al, am, cs, &x, be, &mut y).unwrap(),
        2 => l2::cgemv(Trans::ConjTrans, N, N, al, am, cs, &x, be, &mut y).unwrap(),
        3 => l2::cgeru(N, N, al, &x, &x, am, cs).unwrap(),
        4 => l2::cgerc(N, N, al, &x, &x, am, cs).unwrap(),
        5 => l2::chemv(Uplo::Upper, N, al, am, cs, &x, be, &mut y).unwrap(),
        6 => {
            l1::ccopy(&x, &mut y).unwrap();
            l2::ctrmv(Uplo::Lower, Trans::NoTrans, Diag::NonUnit, N, am, cs, &mut y).unwrap();
        }
        7 => {
            l1::ccopy(&x, &mut y).unwrap();
            l2::ctrsv(Uplo::Lower, Trans::NoTrans, Diag::NonUnit, N, am, cs, &mut y).unwrap();
        }
        8 => l2::cher(Uplo::Upper, N, 0.7, &x, am, cs).unwrap(),
        9 => l2::cher2(Uplo::Upper, N, al, &x, &x, am, cs).unwrap(),
        10 => {
            l1::ccopy(&x, &mut y).unwrap();
            l2::ctrmv(Uplo::Lower, Trans::ConjTrans, Diag::NonUnit, N, am, cs, &mut y).unwrap();
        }
        11 => {
            l1::ccopy(&x, &mut y).unwrap();
            l2::ctrsv(Uplo::Lower, Trans::ConjTrans, Diag::NonUnit, N, am, cs, &mut y).unwrap();
        }
        _ => return f64::NAN,
    }
    let a_probe = &a[..N];
    (l1::scasum(&y) + y[0].re + y[0].im + y[N - 1].re + l1::scasum(a_probe)) as f64
}

/// c32 L3 determinism probes — same op order as run_l3_probe_z.
#[no_mangle]
pub extern "C" fn run_l3_probe_c(op: usize) -> f64 {
    use blas_wasm::L1 as l1;
    use blas_wasm::L3 as l3;
    const N: usize = 65;
    const K: usize = 33;
    let mut st = 13u64;
    let mut next = || {
        st = st
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (((st >> 11) as f64 / (1u64 << 53) as f64) * 2.0 - 1.0) as f32
    };
    let mut a = alloc_mat_c32(N, &mut next);
    for i in 0..N {
        a[i * N + i] = C32::new(2.0 * N as f32 + 1.0, 0.7);
    }
    let b = alloc_mat_c32(N, &mut next);
    let mut c = alloc_mat_c32(N, &mut next);
    let cs = N;
    let len = cs * (N - 1) + N;
    let av = &a[..len];
    let bv = &b[..len];
    let cv = &mut c[..len];
    let al = C32::new(0.7, -0.2);
    let be = C32::new(0.3, 0.1);
    match op {
        0 => l3::cgemm(Trans::NoTrans, Trans::NoTrans, N, N, K, al, av, cs, bv, cs, be, cv, cs, &mut vec![C32::ZERO; l3::cgemm_work(Trans::NoTrans, Trans::NoTrans, N, N, K)]).unwrap(),
        1 => l3::chemm(Side::Left, Uplo::Upper, N, N, al, av, cs, bv, cs, be, cv, cs).unwrap(),
        2 => l3::cherk(Uplo::Upper, Trans::NoTrans, N, K, 0.7, av, cs, 0.3, cv, cs, &mut vec![C32::ZERO; l3::cherk_work(Trans::NoTrans, N, K)]).unwrap(),
        3 => l3::cher2k(Uplo::Upper, Trans::NoTrans, N, K, al, av, cs, bv, cs, 0.3, cv, cs, &mut vec![C32::ZERO; l3::cher2k_work(Trans::NoTrans, N, K)]).unwrap(),
        4 => l3::ctrmm(Side::Left, Uplo::Lower, Trans::NoTrans, Diag::NonUnit, N, N, al, av, cs, cv, cs, &mut vec![C32::ZERO; l3::ctrmm_work(Side::Left, Trans::NoTrans, N, N)]).unwrap(),
        5 => l3::ctrsm(Side::Left, Uplo::Lower, Trans::NoTrans, Diag::NonUnit, N, N, al, av, cs, cv, cs, &mut vec![C32::ZERO; l3::ctrsm_work(Side::Left, Trans::NoTrans, N, N)]).unwrap(),
        6 => l3::ctrmm(Side::Right, Uplo::Upper, Trans::NoTrans, Diag::NonUnit, N, N, al, av, cs, cv, cs, &mut vec![C32::ZERO; l3::ctrmm_work(Side::Right, Trans::NoTrans, N, N)]).unwrap(),
        7 => l3::ctrsm(Side::Right, Uplo::Upper, Trans::NoTrans, Diag::NonUnit, N, N, al, av, cs, cv, cs, &mut vec![C32::ZERO; l3::ctrsm_work(Side::Right, Trans::NoTrans, N, N)]).unwrap(),
        8 => l3::chemm(Side::Right, Uplo::Upper, N, N, al, av, cs, bv, cs, be, cv, cs).unwrap(),
        // the transposed forms (todos S1): the copy zone, the packed
        // transposed read (a shape above this type's packed threshold,
        // its own fold), the syrk family, trmm and trsm on both sides
        9 => l3::cgemm(Trans::ConjTrans, Trans::NoTrans, N, N, K, al, av, cs, bv, cs, be, cv, cs, &mut vec![C32::ZERO; l3::cgemm_work(Trans::ConjTrans, Trans::NoTrans, N, N, K)]).unwrap(),
        10 => {
            const PK: usize = 1030;
            let ab = alloc_mat_c32(PK, &mut next);
            let bb = alloc_mat_c32(PK, &mut next);
            let mut cb = alloc_mat_c32(PK, &mut next);
            let cb = &mut cb[..PK * 8];
            l3::cgemm(Trans::ConjTrans, Trans::NoTrans, PK, 8, PK, al, &ab, PK, &bb[..PK * 8], PK, be, cb, PK, &mut vec![C32::ZERO; l3::cgemm_work(Trans::ConjTrans, Trans::NoTrans, PK, 8, PK)]).unwrap();
            let mut fold = 0.0f64;
            for j in 0..8 {
                fold += l1::scasum(&cb[j * PK..j * PK + PK]) as f64;
            }
            return fold;
        }
        11 => l3::cherk(Uplo::Upper, Trans::ConjTrans, N, K, 0.7, av, cs, 0.3, cv, cs, &mut vec![C32::ZERO; l3::cherk_work(Trans::ConjTrans, N, K)]).unwrap(),
        12 => l3::ctrmm(Side::Left, Uplo::Lower, Trans::ConjTrans, Diag::NonUnit, N, N, al, av, cs, cv, cs, &mut vec![C32::ZERO; l3::ctrmm_work(Side::Left, Trans::ConjTrans, N, N)]).unwrap(),
        13 => l3::ctrsm(Side::Left, Uplo::Lower, Trans::ConjTrans, Diag::NonUnit, N, N, al, av, cs, cv, cs, &mut vec![C32::ZERO; l3::ctrsm_work(Side::Left, Trans::ConjTrans, N, N)]).unwrap(),
        14 => l3::ctrmm(Side::Right, Uplo::Upper, Trans::ConjTrans, Diag::NonUnit, N, N, al, av, cs, cv, cs, &mut vec![C32::ZERO; l3::ctrmm_work(Side::Right, Trans::ConjTrans, N, N)]).unwrap(),
        15 => l3::ctrsm(Side::Right, Uplo::Upper, Trans::ConjTrans, Diag::NonUnit, N, N, al, av, cs, cv, cs, &mut vec![C32::ZERO; l3::ctrsm_work(Side::Right, Trans::ConjTrans, N, N)]).unwrap(),
        _ => return f64::NAN,
    }
    let mut fold = 0.0f32;
    for j in 0..N {
        fold += l1::scasum(&cv[j * cs..j * cs + N]);
    }
    (fold + cv[0].re + cv[0].im + cv[len - 1].re) as f64
}

/// Packed-gemm race rows (tuning campaign): rectangular
/// m×k×n gemm over the square state's buffers reinterpreted as
/// contiguous column-major panels (A = first m·k of a, B = first k·n
/// of b, C = first m·n of sym). Caller must `setup(N)` with
/// m·k, k·n, m·n ≤ N². `ty`: 0 = f64, 1 = f32, 2 = c64, 3 = c32.
/// Same α/β constants as the layer rows. Returns the C-corner fold —
/// incumbent and packed are bit-identical by construction, so the
/// race script asserts the two folds match exactly before timing.
/// Which gemm shape `gemm_rect` times: the netlib wrapper as shipped
/// (the B0 rows), the 4-column fused shape the packed race replaced
/// above the threshold, or the packed-panel shape itself.
#[derive(Clone, Copy)]
enum GemmShape {
    Shipped,
    Col4,
    Tiled,
    Packed,
    PackedBcast,
}

fn gemm_rect(ty: usize, m: usize, k: usize, n: usize, shape: GemmShape) -> f64 {
    use blas_wasm::L3 as l3;
    let s = state_mut();
    let cap = s.n * s.n;
    assert!(m * k <= cap && k * n <= cap && m * n <= cap, "gemm_rect: setup too small");
    let nt = Trans::NoTrans;
    match ty {
        0 => {
            let a = &s.a[..m * k];
            let b = &s.b[..k * n];
            let c = &mut s.sym[..m * n];
            match shape {
                GemmShape::Shipped => l3::dgemm(nt, nt, m, n, k, 0.001, a, m, b, k, 0.5, c, m, ws(&mut s.work, l3::dgemm_work(nt, nt, m, n, k), 0.0)).unwrap(),
                GemmShape::Col4 => l3::dgemm_col4(0.001, m, n, k, a, m, b, k, 0.5, c, m),
                GemmShape::Tiled => l3::dgemm_tiled(0.001, m, n, k, a, m, b, k, 0.5, c, m),
                GemmShape::Packed => l3::dgemm_packed(0.001, m, n, k, a, m, b, k, 0.5, c, m, ws(&mut s.work, l3::dgemm_packed_work(m, n, k), 0.0)),
                GemmShape::PackedBcast => l3::dgemm_packed_bcast(0.001, m, n, k, a, m, b, k, 0.5, c, m, ws(&mut s.work, l3::dgemm_packed_work(m, n, k), 0.0)),
            }
            c[0] + c[m * n - 1]
        }
        1 => {
            let a = &s.a32[..m * k];
            let b = &s.b32[..k * n];
            let c = &mut s.sym32[..m * n];
            match shape {
                GemmShape::Shipped => l3::sgemm(nt, nt, m, n, k, 0.001, a, m, b, k, 0.5, c, m, ws(&mut s.work32, l3::sgemm_work(nt, nt, m, n, k), 0.0)).unwrap(),
                GemmShape::Col4 => l3::sgemm_col4(0.001, m, n, k, a, m, b, k, 0.5, c, m),
                GemmShape::Tiled => l3::sgemm_tiled(0.001, m, n, k, a, m, b, k, 0.5, c, m),
                GemmShape::Packed => l3::sgemm_packed(0.001, m, n, k, a, m, b, k, 0.5, c, m, ws(&mut s.work32, l3::sgemm_packed_work(m, n, k), 0.0)),
                GemmShape::PackedBcast => l3::sgemm_packed_bcast(0.001, m, n, k, a, m, b, k, 0.5, c, m, ws(&mut s.work32, l3::sgemm_packed_work(m, n, k), 0.0)),
            }
            (c[0] + c[m * n - 1]) as f64
        }
        2 => {
            let al = C64::new(0.001, 0.0);
            let be = C64::new(0.5, 0.0);
            let a = &s.az[..m * k];
            let b = &s.bz[..k * n];
            let c = &mut s.symz[..m * n];
            match shape {
                GemmShape::Shipped => l3::zgemm(nt, nt, m, n, k, al, a, m, b, k, be, c, m, ws(&mut s.workz, l3::zgemm_work(nt, nt, m, n, k), C64::ZERO)).unwrap(),
                GemmShape::Col4 => l3::zgemm_col4(al, m, n, k, a, m, b, k, be, c, m),
                GemmShape::Tiled => return f64::NAN, // the complex types have no unpacked tile
                GemmShape::Packed => l3::zgemm_packed(al, m, n, k, a, m, b, k, be, c, m, ws(&mut s.workz, l3::zgemm_packed_work(m, n, k), C64::ZERO)),
                GemmShape::PackedBcast => return f64::NAN, // no complex candidate: the complex kernels already fold α and broadcast at pack time
            }
            c[0].re + c[0].im + c[m * n - 1].re + c[m * n - 1].im
        }
        3 => {
            let al = C32::new(0.001, 0.0);
            let be = C32::new(0.5, 0.0);
            let a = &s.ac[..m * k];
            let b = &s.bc[..k * n];
            let c = &mut s.symc[..m * n];
            match shape {
                GemmShape::Shipped => l3::cgemm(nt, nt, m, n, k, al, a, m, b, k, be, c, m, ws(&mut s.workc, l3::cgemm_work(nt, nt, m, n, k), C32::ZERO)).unwrap(),
                GemmShape::Col4 => l3::cgemm_col4(al, m, n, k, a, m, b, k, be, c, m),
                GemmShape::Tiled => return f64::NAN, // the complex types have no unpacked tile
                GemmShape::Packed => l3::cgemm_packed(al, m, n, k, a, m, b, k, be, c, m, ws(&mut s.workc, l3::cgemm_packed_work(m, n, k), C32::ZERO)),
                GemmShape::PackedBcast => return f64::NAN, // no complex candidate: the complex kernels already fold α and broadcast at pack time
            }
            (c[0].re + c[0].im + c[m * n - 1].re + c[m * n - 1].im) as f64
        }
        _ => f64::NAN,
    }
}

/// The shipped wrapper (the B0 rows); the workspace comes from the setup
/// state, grown on the warmup call, so no allocation lands inside a timed call.
#[no_mangle]
pub extern "C" fn run_gemm_rect_incumbent(ty: usize, m: usize, k: usize, n: usize) -> f64 {
    gemm_rect(ty, m, k, n, GemmShape::Shipped)
}

/// The 4-column fused shape the packed race replaced above the threshold.
#[no_mangle]
pub extern "C" fn run_gemm_rect_col4(ty: usize, m: usize, k: usize, n: usize) -> f64 {
    gemm_rect(ty, m, k, n, GemmShape::Col4)
}

#[no_mangle]
pub extern "C" fn run_gemm_rect_packed(ty: usize, m: usize, k: usize, n: usize) -> f64 {
    gemm_rect(ty, m, k, n, GemmShape::Packed)
}

/// The unpacked register tile (real types only), the shape below the
/// packed threshold.
#[no_mangle]
pub extern "C" fn run_gemm_rect_tiled(ty: usize, m: usize, k: usize, n: usize) -> f64 {
    gemm_rect(ty, m, k, n, GemmShape::Tiled)
}

/// K1: dsyrk/ssyrk at n×n with k columns, the shipping sweep (`packed`
/// = 0) or the packed candidate (1); upper when `upper` != 0. β = 0.5,
/// α = 0.001, C refreshed from the setup state's symmetric buffer; the
/// workspace from the setup state.
#[no_mangle]
pub extern "C" fn run_syrk_shape(ty: usize, n: usize, k: usize, upper: usize, packed: usize) -> f64 {
    use blas_wasm::L3 as l3;
    let s = state_mut();
    let cap = s.n * s.n;
    assert!(n * k <= cap && n * n <= cap, "run_syrk_shape: setup too small");
    let up = upper != 0;
    match ty {
        0 => {
            let a = &s.a[..n * k];
            let c = &mut s.sym[..n * n];
            if packed != 0 {
                l3::dsyrk_packed(up, 0.001, n, k, a, n, 0.5, c, n, ws(&mut s.work, l3::dsyrk_packed_work(n, k), 0.0));
            } else {
                l3::dsyrk(if up { Uplo::Upper } else { Uplo::Lower }, Trans::NoTrans, n, k, 0.001, a, n, 0.5, c, n, ws(&mut s.work, l3::dsyrk_work(Trans::NoTrans, n, k), 0.0)).unwrap();
            }
            c[0] + c[(n - 1) * n + n - 1] + c[if up { (n - 1) * n } else { n - 1 }]
        }
        1 => {
            let a = &s.a32[..n * k];
            let c = &mut s.sym32[..n * n];
            if packed != 0 {
                l3::ssyrk_packed(up, 0.001, n, k, a, n, 0.5, c, n, ws(&mut s.work32, l3::ssyrk_packed_work(n, k), 0.0));
            } else {
                l3::ssyrk(if up { Uplo::Upper } else { Uplo::Lower }, Trans::NoTrans, n, k, 0.001, a, n, 0.5, c, n, ws(&mut s.work32, l3::ssyrk_work(Trans::NoTrans, n, k), 0.0)).unwrap();
            }
            (c[0] + c[(n - 1) * n + n - 1] + c[if up { (n - 1) * n } else { n - 1 }]) as f64
        }
        _ => f64::NAN,
    }
}

/// K1: the triangular routines at m×n (A is m×m on the left, n×n on
/// the right), the shipping kernel (`packed` = 0) or the packed
/// candidate (1); `routine` 0 = trmm, 1 = trsm; `side` 0 = left, 1 =
/// right. B is refreshed from the setup state's source buffer each
/// call; α = 1; the workspace from the setup state. NaN where no
/// candidate exists yet.
#[no_mangle]
pub extern "C" fn run_tri_shape(ty: usize, routine: usize, side: usize, m: usize, n: usize, upper: usize, unit: usize, packed: usize) -> f64 {
    use blas_wasm::L3 as l3;
    let s = state_mut();
    let cap = s.n * s.n;
    let na = if side == 0 { m } else { n };
    assert!(na <= s.n && m * n <= cap, "run_tri_shape: setup too small");
    // A is the na×na leading block of the dominant-diagonal buffer at its own
    // leading dimension, so the solves stay bounded
    let lda = s.n;
    let (up, un) = (upper != 0, unit != 0);
    let (sd, ul, dg) = (if side == 0 { Side::Left } else { Side::Right }, if up { Uplo::Upper } else { Uplo::Lower }, if un { Diag::Unit } else { Diag::NonUnit });
    match ty {
        0 => {
            let a = &s.tri[..];
            let b = &mut s.b[..m * n];
            b.copy_from_slice(&s.sym[..m * n]);
            match (routine, side, packed) {
                (0, _, 0) => l3::dtrmm(sd, ul, Trans::NoTrans, dg, m, n, 1.0, a, lda, b, m, ws(&mut s.work, l3::dtrmm_work(sd, Trans::NoTrans, m, n), 0.0)).unwrap(),
                (1, _, 0) => l3::dtrsm(sd, ul, Trans::NoTrans, dg, m, n, 1.0, a, lda, b, m, ws(&mut s.work, l3::dtrsm_work(sd, Trans::NoTrans, m, n), 0.0)).unwrap(),
                (0, 0, _) => l3::dtrmm_left_packed(up, un, 1.0, m, n, a, lda, b, m, ws(&mut s.work, l3::dtrmm_left_packed_work(m, n), 0.0)),
                (1, 0, _) => l3::dtrsm_left_packed(up, un, 1.0, m, n, a, lda, b, m, ws(&mut s.work, l3::dtrsm_left_packed_work(m, n), 0.0)),
                (0, 1, _) => l3::dtrmm_right_packed(up, un, 1.0, m, n, a, lda, b, m, ws(&mut s.work, l3::dtrmm_right_packed_work(m, n), 0.0)),
                (1, 1, _) => l3::dtrsm_right_packed(up, un, 1.0, m, n, a, lda, b, m, ws(&mut s.work, l3::dtrsm_right_packed_work(m, n), 0.0)),
                _ => return f64::NAN,
            }
            b[0] + b[m * n - 1]
        }
        1 => {
            let a = &s.tri32[..];
            let b = &mut s.b32[..m * n];
            b.copy_from_slice(&s.sym32[..m * n]);
            match (routine, side, packed) {
                (0, _, 0) => l3::strmm(sd, ul, Trans::NoTrans, dg, m, n, 1.0, a, lda, b, m, ws(&mut s.work32, l3::strmm_work(sd, Trans::NoTrans, m, n), 0.0)).unwrap(),
                (1, _, 0) => l3::strsm(sd, ul, Trans::NoTrans, dg, m, n, 1.0, a, lda, b, m, ws(&mut s.work32, l3::strsm_work(sd, Trans::NoTrans, m, n), 0.0)).unwrap(),
                (0, 0, _) => l3::strmm_left_packed(up, un, 1.0, m, n, a, lda, b, m, ws(&mut s.work32, l3::strmm_left_packed_work(m, n), 0.0)),
                (1, 0, _) => l3::strsm_left_packed(up, un, 1.0, m, n, a, lda, b, m, ws(&mut s.work32, l3::strsm_left_packed_work(m, n), 0.0)),
                (0, 1, _) => l3::strmm_right_packed(up, un, 1.0, m, n, a, lda, b, m, ws(&mut s.work32, l3::strmm_right_packed_work(m, n), 0.0)),
                (1, 1, _) => l3::strsm_right_packed(up, un, 1.0, m, n, a, lda, b, m, ws(&mut s.work32, l3::strsm_right_packed_work(m, n), 0.0)),
                _ => return f64::NAN,
            }
            (b[0] + b[m * n - 1]) as f64
        }
        _ => f64::NAN,
    }
}

/// The K2 candidate (real types only).
#[no_mangle]
pub extern "C" fn run_gemm_rect_packed_bcast(ty: usize, m: usize, k: usize, n: usize) -> f64 {
    gemm_rect(ty, m, k, n, GemmShape::PackedBcast)
}

/// Transposed trmm/trsm race (STATUS.md, decision D3): the two
/// candidates for op(A) = Aᵀ (Aᴴ for complex), `copy` (transpose into
/// workspace, existing kernel) and `stream` (no copy; transposed
/// accessor on the right, column-reduction kernels on the left).
/// `routine` 0 = trmm, 1 = trsm; `side` 0 = left, 1 = right; `cand`
/// 0 = copy, 1 = stream. A is the diagonally dominant `tri` buffer at
/// the setup leading dimension, so the solves stay bounded; B is
/// refilled from `b` before every call (same cost on both sides) so
/// repeated multiplies cannot overflow; the copy's workspace comes from
/// the setup state, grown on the warmup call, so neither side allocates
/// inside a timed call. Returns a fold of B; the two
/// candidates are bit-identical by construction, so the race script
/// asserts the folds match before timing.
fn tri_t(ty: usize, routine: usize, side: usize, upper: bool, m: usize, n: usize, cand: usize) -> f64 {
    use blas_wasm::L3 as l3;
    let s = state_mut();
    let na = if side == 0 { m } else { n };
    assert!(na <= s.n && m * n <= s.n * s.n, "tri_t: setup too small");
    let lda = s.n;
    let sd = if side == 0 { Side::Left } else { Side::Right };
    let up = if upper { Uplo::Upper } else { Uplo::Lower };
    let dg = Diag::NonUnit;
    macro_rules! run {
        ($a:expr, $b:expr, $src:expr, $w:expr, $one:expr, $ta:expr, $zero:expr, $trmm_copy:ident, $trmm_copy_work:ident, $trmm_stream:ident, $trsm_copy:ident, $trsm_copy_work:ident, $trsm_stream:ident) => {{
            let a = &$a[..];
            let b = &mut $b[..m * n];
            b.copy_from_slice(&$src[..m * n]);
            let alpha = $one;
            match (routine, cand) {
                (0, 0) => {
                    let w = ws(&mut $w, l3::$trmm_copy_work(sd, m, n), $zero);
                    l3::$trmm_copy(sd, up, $ta, dg, m, n, alpha, a, lda, b, m, w);
                }
                (0, _) => l3::$trmm_stream(sd, up, $ta, dg, m, n, alpha, a, lda, b, m),
                (_, 0) => {
                    let w = ws(&mut $w, l3::$trsm_copy_work(sd, m, n), $zero);
                    l3::$trsm_copy(sd, up, $ta, dg, m, n, alpha, a, lda, b, m, w);
                }
                (_, _) => l3::$trsm_stream(sd, up, $ta, dg, m, n, alpha, a, lda, b, m),
            }
            b
        }};
    }
    match ty {
        0 => {
            let b = run!(s.tri, s.sym, s.b, s.work, 1.0f64, Trans::Trans, 0.0f64, dtrmm_copy, dtrmm_copy_work, dtrmm_stream, dtrsm_copy, dtrsm_copy_work, dtrsm_stream);
            b[0] + b[m * n - 1]
        }
        1 => {
            let b = run!(s.tri32, s.sym32, s.b32, s.work32, 1.0f32, Trans::Trans, 0.0f32, strmm_copy, strmm_copy_work, strmm_stream, strsm_copy, strsm_copy_work, strsm_stream);
            (b[0] + b[m * n - 1]) as f64
        }
        2 => {
            let b = run!(s.triz, s.symz, s.bz, s.workz, C64::ONE, Trans::ConjTrans, C64::ZERO, ztrmm_copy, ztrmm_copy_work, ztrmm_stream, ztrsm_copy, ztrsm_copy_work, ztrsm_stream);
            b[0].re + b[0].im + b[m * n - 1].re + b[m * n - 1].im
        }
        _ => {
            let b = run!(s.tric, s.symc, s.bc, s.workc, C32::ONE, Trans::ConjTrans, C32::ZERO, ctrmm_copy, ctrmm_copy_work, ctrmm_stream, ctrsm_copy, ctrsm_copy_work, ctrsm_stream);
            (b[0].re + b[0].im + b[m * n - 1].re + b[m * n - 1].im) as f64
        }
    }
}

#[no_mangle]
pub extern "C" fn run_tri_t_copy(ty: usize, routine: usize, side: usize, upper: usize, m: usize, n: usize) -> f64 {
    tri_t(ty, routine, side, upper != 0, m, n, 0)
}

#[no_mangle]
pub extern "C" fn run_tri_t_stream(ty: usize, routine: usize, side: usize, upper: usize, m: usize, n: usize) -> f64 {
    tri_t(ty, routine, side, upper != 0, m, n, 1)
}

/// Transposed gemm race (STATUS.md, decision D3 applied to gemm): the
/// two candidates for op(A)/op(B), `copy` (`*gemm_copy_op`: the
/// transposed operand copied into the workspace, then the size-dispatched
/// shape) and `pack` (`*gemm_packed_op`: the packed shape packing from a
/// transposed read); both are called directly, not through the netlib
/// wrapper, so the race stays a race after the wrapper routes a winner.
/// Workspaces come from the setup state, grown on the warmup call. `transa`/`transb` 0 = N, 1 = T, 2 = C. A and B are
/// the setup buffers read at the stored shape's leading dimension; C is
/// the sacrificial buffer with β = 0, so the fold does not depend on the
/// previous call. Returns a fold of C; above the
/// packed threshold both candidates run the packed shape and are
/// bit-identical by construction, so the race script asserts the folds
/// match before timing.
fn gemm_t(ty: usize, transa: usize, transb: usize, m: usize, k: usize, n: usize, pack: bool) -> f64 {
    use blas_wasm::L3 as l3;
    let s = state_mut();
    let cap = s.n * s.n;
    assert!(m * k <= cap && k * n <= cap && m * n <= cap, "gemm_t: setup too small");
    let tr = |v: usize| match v {
        0 => Trans::NoTrans,
        1 => Trans::Trans,
        _ => Trans::ConjTrans,
    };
    let (ta, tb) = (tr(transa), tr(transb));
    let (ar, ac) = if transa == 0 { (m, k) } else { (k, m) };
    let (br, bc) = if transb == 0 { (k, n) } else { (n, k) };
    macro_rules! run {
        ($a:expr, $b:expr, $c:expr, $w:expr, $al:expr, $be:expr, $zero:expr, $copy_op:ident, $copy_op_work:ident, $packed_op:ident, $packed_op_work:ident) => {{
            let a = &$a[..ar * ac];
            let b = &$b[..br * bc];
            let c = &mut $c[..m * n];
            if pack {
                let w = ws(&mut $w, l3::$packed_op_work(ta, tb, m, n, k), $zero);
                l3::$packed_op(ta, tb, $al, m, n, k, a, ar, b, br, $be, c, m, w);
            } else {
                let w = ws(&mut $w, l3::$copy_op_work(ta, tb, m, n, k), $zero);
                l3::$copy_op(ta, tb, $al, m, n, k, a, ar, b, br, $be, c, m, w);
            }
            c
        }};
    }
    match ty {
        0 => {
            let c = run!(s.a, s.b, s.sym, s.work, 0.001f64, 0.0f64, 0.0f64, dgemm_copy_op, dgemm_copy_op_work, dgemm_packed_op, dgemm_packed_op_work);
            c[0] + c[m * n - 1]
        }
        1 => {
            let c = run!(s.a32, s.b32, s.sym32, s.work32, 0.001f32, 0.0f32, 0.0f32, sgemm_copy_op, sgemm_copy_op_work, sgemm_packed_op, sgemm_packed_op_work);
            (c[0] + c[m * n - 1]) as f64
        }
        2 => {
            let c = run!(s.az, s.bz, s.symz, s.workz, C64::new(0.001, 0.0), C64::ZERO, C64::ZERO, zgemm_copy_op, zgemm_copy_op_work, zgemm_packed_op, zgemm_packed_op_work);
            c[0].re + c[0].im + c[m * n - 1].re + c[m * n - 1].im
        }
        _ => {
            let c = run!(s.ac, s.bc, s.symc, s.workc, C32::new(0.001, 0.0), C32::ZERO, C32::ZERO, cgemm_copy_op, cgemm_copy_op_work, cgemm_packed_op, cgemm_packed_op_work);
            (c[0].re + c[0].im + c[m * n - 1].re + c[m * n - 1].im) as f64
        }
    }
}

#[no_mangle]
pub extern "C" fn run_gemm_t_copy(ty: usize, transa: usize, transb: usize, m: usize, k: usize, n: usize) -> f64 {
    gemm_t(ty, transa, transb, m, k, n, false)
}

#[no_mangle]
pub extern "C" fn run_gemm_t_pack(ty: usize, transa: usize, transb: usize, m: usize, k: usize, n: usize) -> f64 {
    gemm_t(ty, transa, transb, m, k, n, true)
}
