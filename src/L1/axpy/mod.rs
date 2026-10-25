#![doc = include_str!("README.md")]

pub mod daxpy;
pub mod saxpy;
pub mod zaxpy;
pub mod caxpy;

pub use daxpy::daxpy;
pub(crate) use daxpy::daxpy_k;
pub use saxpy::saxpy;
pub(crate) use saxpy::saxpy_k;
pub use zaxpy::zaxpy;
pub(crate) use zaxpy::zaxpy_k;
pub use caxpy::caxpy;
pub(crate) use caxpy::caxpy_k;
