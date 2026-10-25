//! Level 1: vector operations, one file per BLAS routine (netlib
//! naming — see README.md in this folder for the convention). The
//! d-prefixed routines are f64, s-prefixed f32, z-prefixed c64,
//! c-prefixed c32; i-prefixed index routines carry the type second
//! (idamax/icamax), and the complex routines returning reals carry
//! both letters (dznrm2/scasum). `drotg`/`srotg`/`zrotg`/`crotg`
//! also export their Givens result structs via their module paths.



pub mod asum;
pub mod axpy;
pub mod copy;
pub mod dot;
pub mod iamax;
pub mod nrm2;
pub mod rot;
pub mod rotg;
pub mod scal;
pub mod swap;

pub use asum::*;
pub use axpy::*;
pub use copy::*;
pub use dot::*;
pub use iamax::*;
pub use nrm2::*;
pub use rot::*;
pub use rotg::*;
pub use scal::*;
pub use swap::*;
