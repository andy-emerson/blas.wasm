# axpy — scaled vector addition

$$ y \leftarrow \alpha x + y $$

for vectors $x, y$ of length $n$ and a scalar $\alpha$.
Reference BLAS names the four type variants `daxpy`, `saxpy`,
`zaxpy` and `caxpy`. This chapter follows `daxpy`; the other three
are the same shape at their own lane width, with the complex pair's
product form documented in their source files.

## What each element is

Element $i$ of the result is, exactly,

$$ y_i \leftarrow \mathrm{fl}\big(y_i + \mathrm{fl}(\alpha\, x_i)\big), $$

one multiply rounding and one add rounding, in that order. There is
no fused multiply-add: a fused form rounds once, and relaxed-SIMD
fused multiply-add rounds differently on different hardware, which
would forfeit the cross-target bit-identity promise. That trade-off is
recorded as a deliberately unused lever in `DESIGN.md`.

## Why the SIMD loop is bit-identical to the definition

The operation is elementwise: no element's result depends on any
other element. A SIMD lane performs the same two roundings on its own
element that the scalar definition does, and lanes never combine. So
the result is independent of the lane width, of how many elements an
iteration processes, and of whether the loop runs on a wasm engine
with simd128 or on native hardware through the lane emulation in
`src/lanes.rs`, which performs the lane operations elementwise. The
tail loop, which handles the last $n \bmod 4$ elements one at a
time, performs the same two roundings. Every target therefore
produces the same bits, and the test for this routine checks exactly
that: bit-for-bit equality against the scalar definition.

## The shipping code

The public entry point checks the lengths and hands the raw pointers
to the SIMD loop. This is the source as it ships, included from
`src/L1/daxpy.rs` at build time:

```rust,ignore
{{#include ../../../src/L1/daxpy.rs:daxpy}}
```

The loop runs two f64 lanes per vector register and is unrolled
twice, so one iteration moves four elements; the `target_feature`
attribute is what makes the intrinsics compile inline on wasm (without
it they compile as out-of-line calls, a measured 6.4× slowdown):

```rust,ignore
{{#include ../../../src/L1/daxpy.rs:daxpy_imp}}
```

## Calling it

This example is compiled and run when the book is built, so it
cannot drift from the crate:

```rust
extern crate blas_wasm;
use blas_wasm::L1::daxpy;

fn main() {
    let x = [1.0, 2.0, 3.0, 4.0, 5.0];
    let mut y = [10.0, 20.0, 30.0, 40.0, 50.0];
    daxpy(2.0, &x, &mut y);
    assert_eq!(y, [12.0, 24.0, 36.0, 48.0, 60.0]);
}
```

## Measured

axpy is a streaming operation and is scored against the fastest
memory stream measured in the same run. The current scoreboard, with
the runs it came from, is `bench/README.md` in the repository; the
per-routine test standard is `tests/README.md`.
