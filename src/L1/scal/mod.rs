#![doc = include_str!("README.md")]

pub mod dscal;
pub mod sscal;
pub mod zscal;
pub mod cscal;
pub mod zdscal;
pub mod csscal;

pub use dscal::dscal;
pub use sscal::sscal;
pub use zscal::zscal;
pub use cscal::cscal;
pub use zdscal::zdscal;
pub use csscal::csscal;
