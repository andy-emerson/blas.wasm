#![doc = include_str!("README.md")]

pub mod dsyrk;
pub mod ssyrk;
pub mod zherk;
pub mod cherk;
pub mod zsyrk;
pub mod csyrk;

pub use dsyrk::{dsyrk, dsyrk_work, dsyrk_packed, dsyrk_packed_work, dsyrk_sweep};
pub use ssyrk::{ssyrk, ssyrk_work, ssyrk_packed, ssyrk_packed_work, ssyrk_sweep};
pub use zherk::{zherk, zherk_work};
pub use cherk::{cherk, cherk_work};
pub use zsyrk::{zsyrk, zsyrk_work};
pub use csyrk::{csyrk, csyrk_work};
