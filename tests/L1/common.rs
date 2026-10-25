//! Level 1 test helpers: the shared generator and summers plus the
//! vector size list.

#![allow(dead_code)] // each per-routine test file uses its own subset

#[path = "../shared.rs"]
mod shared;
pub use shared::*;

// Sizes that exercise the empty case, the pure-tail paths, unroll
// boundaries at both lane widths, and odd tails.
pub const SIZES: &[usize] = &[0, 1, 2, 3, 4, 5, 7, 8, 15, 16, 17, 33, 100, 257, 1000];
