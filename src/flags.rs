//! The flag arguments of the netlib interface as Rust enums, with
//! netlib meanings. The CBLAS projection (todo C in `STATUS.md`) will
//! map each variant onto the standard `CBLAS_*` constant one to one;
//! today's flat module takes `u32` flags.

/// Whether an operand is used as itself, transposed, or conjugate
/// transposed (`TRANS` in netlib: `'N'`, `'T'`, `'C'`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Trans {
	/// The operand as stored: `op(A) = A`.
	NoTrans,
	/// `op(A) = Aᵀ`.
	Trans,
	/// `op(A) = Aᴴ`. For real types this is the same as [`Trans::Trans`].
	ConjTrans,
}

/// Which side a matrix operand multiplies from (`SIDE` in netlib:
/// `'L'`, `'R'`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Side {
	/// `C ← αAB + βC`, `B ← αop(A)B`, `B ← αop(A)⁻¹B`.
	Left,
	/// `C ← αBA + βC`, `B ← αBop(A)`, `B ← αBop(A)⁻¹`.
	Right,
}

/// Which triangle of a symmetric, Hermitian or triangular matrix is
/// stored (`UPLO` in netlib: `'U'`, `'L'`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Uplo {
	/// The upper triangle is referenced; the strict lower triangle is
	/// not read.
	Upper,
	/// The lower triangle is referenced; the strict upper triangle is
	/// not read.
	Lower,
}

/// Whether a triangular matrix has an implicit unit diagonal (`DIAG`
/// in netlib: `'N'`, `'U'`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Diag {
	/// The diagonal is stored and used.
	NonUnit,
	/// The diagonal is taken as ones; the stored diagonal is not read.
	Unit,
}

impl Uplo {
	/// `true` for [`Uplo::Upper`].
	#[inline]
	pub fn is_upper(self) -> bool {
		matches!(self, Uplo::Upper)
	}
}

impl Diag {
	/// `true` for [`Diag::Unit`].
	#[inline]
	pub fn is_unit(self) -> bool {
		matches!(self, Diag::Unit)
	}
}
