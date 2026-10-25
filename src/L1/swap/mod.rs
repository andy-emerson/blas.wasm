#![doc = include_str!("README.md")]

pub mod dswap;
pub mod sswap;
pub mod zswap;
pub mod cswap;

pub use dswap::dswap;
pub(crate) use dswap::dswap_k;
pub use sswap::sswap;
pub(crate) use sswap::sswap_k;
pub use zswap::zswap;
pub use cswap::cswap;
