#![doc = include_str!("README.md")]

pub mod dasum;
pub mod dzasum;
pub mod sasum;
pub mod scasum;

pub use dasum::dasum;
pub use dzasum::dzasum;
pub use sasum::sasum;
pub use scasum::scasum;
