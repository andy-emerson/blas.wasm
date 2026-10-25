#![doc = include_str!("README.md")]

pub mod dsyr2k;
pub mod ssyr2k;
pub mod zher2k;
pub mod cher2k;
pub mod zsyr2k;
pub mod csyr2k;

pub use dsyr2k::{dsyr2k, dsyr2k_work};
pub use ssyr2k::{ssyr2k, ssyr2k_work};
pub use zher2k::{zher2k, zher2k_work};
pub use cher2k::{cher2k, cher2k_work};
pub use zsyr2k::{zsyr2k, zsyr2k_work};
pub use csyr2k::{csyr2k, csyr2k_work};
