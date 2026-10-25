#![doc = include_str!("README.md")]

pub mod dgemv;
pub mod sgemv;
pub mod zgemv;
pub mod cgemv;

pub use dgemv::dgemv;
pub(crate) use dgemv::dgemv_k;
pub use sgemv::sgemv;
pub(crate) use sgemv::sgemv_k;
pub use zgemv::zgemv;
pub(crate) use zgemv::zgemv_k;
pub use cgemv::cgemv;
pub(crate) use cgemv::cgemv_k;
