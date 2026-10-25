#![doc = include_str!("README.md")]

pub mod ddot;
pub mod sdot;
pub mod zdotu;
pub mod zdotc;
pub mod cdotu;
pub mod cdotc;

pub use ddot::ddot;
pub(crate) use ddot::ddot_k;
pub use sdot::sdot;
pub(crate) use sdot::sdot_k;
pub use zdotu::zdotu;
pub(crate) use zdotu::zdotu_k;
pub use zdotc::zdotc;
pub(crate) use zdotc::zdotc_k;
pub use cdotu::cdotu;
pub(crate) use cdotu::cdotu_k;
pub use cdotc::cdotc;
pub(crate) use cdotc::cdotc_k;
