//! Argument errors, reported rather than trapped.
//!
//! Every routine checks what reference BLAS checks before it computes,
//! and a few things reference BLAS cannot check (a slice shorter than
//! the extent its dimensions declare, a workspace shorter than the
//! query says). A failed check returns an [`Error`] naming the routine
//! and the position of the offending argument in this crate's
//! signature, counted from one, the way `xerbla` reports it; nothing
//! has been written when an error is returned. The crate never panics
//! on input. Out-of-range lane access, which would be a bug in a
//! kernel rather than a bad argument, still asserts.

/// Why an argument was rejected.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ErrorKind {
	/// A leading dimension is smaller than the row count it must span
	/// (`lda < max(1, rows)`), or a flag combination the reference
	/// interface forbids (such as `Trans::Trans` on a Hermitian
	/// rank-k update).
	Argument,
	/// A slice is shorter than the extent its dimensions declare.
	Length,
	/// The workspace slice is shorter than the size query for these
	/// dimensions.
	Workspace,
	/// A form the reference interface defines but this crate has not
	/// built yet; which forms are missing is listed in `STATUS.md`.
	Unsupported,
}

/// An argument error from one routine.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Error {
	/// The netlib name of the routine, e.g. `"dgemm"`.
	pub routine: &'static str,
	/// The position of the offending argument in this crate's
	/// signature, counted from one.
	pub arg: u8,
	/// Why it was rejected.
	pub kind: ErrorKind,
}

impl core::fmt::Display for Error {
	fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
		let what = match self.kind {
			ErrorKind::Argument => "invalid value",
			ErrorKind::Length => "slice shorter than its declared extent",
			ErrorKind::Workspace => "workspace shorter than the size query",
			ErrorKind::Unsupported => "form not built yet",
		};
		write!(f, "{}: argument {} has an {}", self.routine, self.arg, what)
	}
}

impl core::error::Error for Error {}

/// A column-major matrix argument: `ld >= max(1, rows)` (reported on
/// `arg_ld`, as reference BLAS requires even for an empty matrix) and
/// `len >= ld·(cols − 1) + rows` when the matrix has elements (reported
/// on `arg`).
#[inline]
pub(crate) fn mat(
	routine: &'static str,
	arg: u8,
	arg_ld: u8,
	len: usize,
	rows: usize,
	cols: usize,
	ld: usize,
) -> Result<(), Error> {
	if ld < rows.max(1) {
		return Err(Error { routine, arg: arg_ld, kind: ErrorKind::Argument });
	}
	if rows > 0 && cols > 0 && len < ld * (cols - 1) + rows {
		return Err(Error { routine, arg, kind: ErrorKind::Length });
	}
	Ok(())
}

/// A contiguous vector argument: `len >= n`.
#[inline]
pub(crate) fn vec(routine: &'static str, arg: u8, len: usize, n: usize) -> Result<(), Error> {
	if len < n {
		return Err(Error { routine, arg, kind: ErrorKind::Length });
	}
	Ok(())
}

/// Two vectors of the same length (the Level 1 two-vector routines).
#[inline]
pub(crate) fn same_len(routine: &'static str, arg: u8, a: usize, b: usize) -> Result<(), Error> {
	if a != b {
		return Err(Error { routine, arg, kind: ErrorKind::Length });
	}
	Ok(())
}

/// The workspace slice against the size query.
#[inline]
pub(crate) fn work(routine: &'static str, arg: u8, len: usize, need: usize) -> Result<(), Error> {
	if len < need {
		return Err(Error { routine, arg, kind: ErrorKind::Workspace });
	}
	Ok(())
}

#[inline]
pub(crate) fn unsupported(routine: &'static str, arg: u8) -> Error {
	Error { routine, arg, kind: ErrorKind::Unsupported }
}

#[inline]
pub(crate) fn argument(routine: &'static str, arg: u8) -> Error {
	Error { routine, arg, kind: ErrorKind::Argument }
}
