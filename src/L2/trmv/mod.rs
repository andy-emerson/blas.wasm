#![doc = include_str!("README.md")]

pub mod dtrmv;
pub mod strmv;
pub mod ztrmv;
pub mod ctrmv;

pub use dtrmv::dtrmv;
pub(crate) use dtrmv::dtrmv_k;
pub use strmv::strmv;
pub(crate) use strmv::strmv_k;
pub use ztrmv::ztrmv;
pub(crate) use ztrmv::ztrmv_k;
pub use ctrmv::ctrmv;
pub(crate) use ctrmv::ctrmv_k;
