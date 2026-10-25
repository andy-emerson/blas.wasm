#![doc = include_str!("README.md")]

pub mod dtrmm;
pub mod strmm;
pub mod ztrmm;
pub mod ctrmm;

pub use dtrmm::{dtrmm, dtrmm_work, dtrmm_copy, dtrmm_copy_work, dtrmm_stream, dtrmm_left_packed, dtrmm_left_packed_work, dtrmm_right_packed, dtrmm_right_packed_work, dtrmm_left_sweep, dtrmm_right_sweep};
pub use strmm::{strmm, strmm_work, strmm_copy, strmm_copy_work, strmm_stream, strmm_left_packed, strmm_left_packed_work, strmm_right_packed, strmm_right_packed_work, strmm_left_sweep, strmm_right_sweep};
pub use ztrmm::{ztrmm, ztrmm_work, ztrmm_copy, ztrmm_copy_work, ztrmm_stream};
pub use ctrmm::{ctrmm, ctrmm_work, ctrmm_copy, ctrmm_copy_work, ctrmm_stream};
