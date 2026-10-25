#![doc = include_str!("README.md")]

pub mod dcopy;
pub mod scopy;
pub mod zcopy;
pub mod ccopy;

pub use dcopy::dcopy;
pub(crate) use dcopy::dcopy_k;
pub use scopy::scopy;
pub(crate) use scopy::scopy_k;
pub use zcopy::zcopy;
pub use ccopy::ccopy;
