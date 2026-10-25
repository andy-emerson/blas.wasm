#![doc = include_str!("README.md")]

pub mod dtrsm;
pub mod strsm;
pub mod ztrsm;
pub mod ctrsm;

pub use dtrsm::{dtrsm, dtrsm_work, dtrsm_copy, dtrsm_copy_work, dtrsm_stream, dtrsm_left_packed, dtrsm_left_packed_work, dtrsm_right_packed, dtrsm_right_packed_work, dtrsm_left_sweep, dtrsm_right_sweep};
pub use strsm::{strsm, strsm_work, strsm_copy, strsm_copy_work, strsm_stream, strsm_left_packed, strsm_left_packed_work, strsm_right_packed, strsm_right_packed_work, strsm_left_sweep, strsm_right_sweep};
pub use ztrsm::{ztrsm, ztrsm_work, ztrsm_copy, ztrsm_copy_work, ztrsm_stream};
pub use ctrsm::{ctrsm, ctrsm_work, ctrsm_copy, ctrsm_copy_work, ctrsm_stream};
