//! Level 1: vector operations, one file per BLAS routine (netlib
//! naming — see README.md in this folder for the convention). The
//! d-prefixed routines are f64, s-prefixed f32, z-prefixed c64,
//! c-prefixed c32; i-prefixed index routines carry the type second
//! (idamax/icamax), and the complex routines returning reals carry
//! both letters (dznrm2/scasum). `drotg`/`srotg`/`zrotg`/`crotg`
//! also export their Givens result structs via their module paths.

pub mod dasum;
pub mod daxpy;
pub mod dcopy;
pub mod ddot;
pub mod dnrm2;
pub mod drot;
pub mod drotg;
pub mod dscal;
pub mod dswap;
pub mod idamax;
pub mod isamax;
pub mod sasum;
pub mod saxpy;
pub mod scopy;
pub mod sdot;
pub mod snrm2;
pub mod srot;
pub mod srotg;
pub mod sscal;
pub mod sswap;
pub mod dzasum;
pub mod dznrm2;
pub mod izamax;
pub mod zaxpy;
pub mod zcopy;
pub mod zdotc;
pub mod zdotu;
pub mod zdrot;
pub mod zrot;
pub mod zdscal;
pub mod zrotg;
pub mod zscal;
pub mod zswap;
pub mod caxpy;
pub mod ccopy;
pub mod cdotc;
pub mod cdotu;
pub mod crotg;
pub mod cscal;
pub mod csrot;
pub mod crot;
pub mod csscal;
pub mod cswap;
pub mod icamax;
pub mod scasum;
pub mod scnrm2;

pub use dasum::dasum;
pub use daxpy::daxpy;
#[allow(unused_imports)]
pub(crate) use daxpy::daxpy_k;
pub use dcopy::dcopy;
#[allow(unused_imports)]
pub(crate) use dcopy::dcopy_k;
pub use ddot::ddot;
#[allow(unused_imports)]
pub(crate) use ddot::ddot_k;
pub use dnrm2::dnrm2;
pub use drot::drot;
#[allow(unused_imports)]
pub(crate) use drot::drot_k;
pub use drotg::drotg;
pub use dscal::dscal;
pub use dswap::dswap;
#[allow(unused_imports)]
pub(crate) use dswap::dswap_k;
pub use idamax::idamax;
pub use isamax::isamax;
pub use sasum::sasum;
pub use saxpy::saxpy;
#[allow(unused_imports)]
pub(crate) use saxpy::saxpy_k;
pub use scopy::scopy;
#[allow(unused_imports)]
pub(crate) use scopy::scopy_k;
pub use sdot::sdot;
#[allow(unused_imports)]
pub(crate) use sdot::sdot_k;
pub use snrm2::snrm2;
pub use srot::srot;
#[allow(unused_imports)]
pub(crate) use srot::srot_k;
pub use srotg::srotg;
pub use sscal::sscal;
pub use sswap::sswap;
#[allow(unused_imports)]
pub(crate) use sswap::sswap_k;
pub use dzasum::dzasum;
pub use dznrm2::dznrm2;
pub use izamax::izamax;
pub use zaxpy::zaxpy;
#[allow(unused_imports)]
pub(crate) use zaxpy::zaxpy_k;
pub use zcopy::zcopy;
#[allow(unused_imports)]
pub(crate) use zcopy::zcopy_k;
pub use zdotc::zdotc;
#[allow(unused_imports)]
pub(crate) use zdotc::zdotc_k;
pub use zdotu::zdotu;
#[allow(unused_imports)]
pub(crate) use zdotu::zdotu_k;
pub use zdrot::zdrot;
pub use zrot::zrot;
#[allow(unused_imports)]
pub(crate) use zdrot::zdrot_k;
pub use zdscal::zdscal;
pub use zrotg::zrotg;
pub use zscal::zscal;
pub use zswap::zswap;
#[allow(unused_imports)]
pub(crate) use zswap::zswap_k;
pub use caxpy::caxpy;
#[allow(unused_imports)]
pub(crate) use caxpy::caxpy_k;
pub use ccopy::ccopy;
#[allow(unused_imports)]
pub(crate) use ccopy::ccopy_k;
pub use cdotc::cdotc;
#[allow(unused_imports)]
pub(crate) use cdotc::cdotc_k;
pub use cdotu::cdotu;
#[allow(unused_imports)]
pub(crate) use cdotu::cdotu_k;
pub use crotg::crotg;
pub use cscal::cscal;
pub use csrot::csrot;
pub use crot::crot;
#[allow(unused_imports)]
pub(crate) use csrot::csrot_k;
pub use csscal::csscal;
pub use cswap::cswap;
#[allow(unused_imports)]
pub(crate) use cswap::cswap_k;
pub use icamax::icamax;
pub use scasum::scasum;
pub use scnrm2::scnrm2;
