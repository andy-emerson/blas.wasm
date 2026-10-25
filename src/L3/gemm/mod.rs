#![doc = include_str!("README.md")]

pub mod dgemm;
pub mod sgemm;
pub mod zgemm;
pub mod cgemm;

pub use dgemm::{dgemm, dgemm_work, dgemm_packed_work, dgemm_packed_op, dgemm_packed_op_work, dgemm_copy_op, dgemm_copy_op_work, dgemm_colaxpy, dgemm_tiled, dgemm_col4, dgemm_packed, dgemm_packed_bcast};
pub use sgemm::{sgemm, sgemm_work, sgemm_packed_work, sgemm_packed_op, sgemm_packed_op_work, sgemm_copy_op, sgemm_copy_op_work, sgemm_colaxpy, sgemm_tiled, sgemm_col4, sgemm_packed, sgemm_packed_bcast};
pub use zgemm::{zgemm, zgemm_work, zgemm_packed_work, zgemm_packed_op, zgemm_packed_op_work, zgemm_copy_op, zgemm_copy_op_work, zgemm_colaxpy, zgemm_col4, zgemm_packed};
pub use cgemm::{cgemm, cgemm_work, cgemm_packed_work, cgemm_packed_op, cgemm_packed_op_work, cgemm_copy_op, cgemm_copy_op_work, cgemm_colaxpy, cgemm_col4, cgemm_packed};
