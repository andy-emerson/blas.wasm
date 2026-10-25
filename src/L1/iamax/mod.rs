#![doc = include_str!("README.md")]

pub mod idamax;
pub mod isamax;
pub mod izamax;
pub mod icamax;

pub use idamax::idamax;
pub use isamax::isamax;
pub use izamax::izamax;
pub use icamax::icamax;
