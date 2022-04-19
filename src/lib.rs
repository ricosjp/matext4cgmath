//! Unofficial third-party cgmath extensions for calculate eigenvalues, operator norms and Iwasawa decomposition.

#![cfg_attr(not(debug_assertions), deny(warnings))]
#![deny(clippy::all, rust_2018_idioms)]
#![warn(
    missing_docs,
    missing_debug_implementations,
    trivial_casts,
    trivial_numeric_casts,
    unsafe_code,
    unstable_features,
    unused_import_braces,
    unused_qualifications
)]
#![allow(clippy::many_single_char_names)]

pub use cgmath;
use cgmath::*;
use num_complex::Complex;

mod eigens;
mod exp_decomp;
/// solvers for low dimensional algebraic equations.
pub mod solver;

/// extension for eigen values
pub trait EigenValues: VectorSpace {
    /// the type of the array of eigen values
    type EigenValues;
    /// calculate eigen values.
    ///
    /// # Examples
    ///
    /// ```
    /// use cgmath::*;
    /// use cgmath_matrix_extensions::*;
    /// use num_complex::Complex;
    /// const EPS: f64 = 1.0e-10;
    ///
    /// let mat = Matrix2::new(4.0, -2.0, 3.0, -1.0);
    /// let mut eigens = mat.eigenvalues();
    /// // Even in the case of real solutions, the order in the array is not guaranteed.
    /// eigens.sort_by(|x, y| x.re.partial_cmp(&y.re).unwrap());
    /// assert!(Complex::norm(eigens[0] - 1.0) < EPS);
    /// assert!(Complex::norm(eigens[1] - 2.0) < EPS);
    /// ```
    fn eigenvalues(self) -> Self::EigenValues;
}

/// operator norms: abs sum, Euclid, and max.
pub trait OperatorNorm: VectorSpace {
    /// operator norm for absolute value sumation: $l^1$.
    fn norm_l1(self) -> Self::Scalar;
    /// operator norm for Euclidean norm.
    fn norm_l2(self) -> Self::Scalar;
    /// operator norm for absolute value maximum: $l^{\infty}$.
    fn norm_linf(self) -> Self::Scalar;
}

/// calculate exponential value
pub trait Exponential: OperatorNorm + std::ops::AddAssign<Self> + One
where
    Self::Scalar: BaseFloat,
{
    /// calculate exponential
    fn exp(self) -> Self {
        use num_traits::{Float, NumCast};
        let eps = <Self::Scalar as Float>::epsilon();
        let mut a = self;
        let mut res = Self::one();
        for i in 2_u16..=64 {
            res += a;
            a = a * self / <Self::Scalar as NumCast>::from(i).unwrap();
            if a.norm_linf() < eps * self.norm_linf() {
                break;
            }
        }
        res
    }
}

/// some decompositions of matrix
pub trait Decomposition: VectorSpace {
    /// Returns $(K, S)$ of the Cartan decomposition: $M = K exp(S)$.
    ///
    /// - $K$: orthonormal matrix
    /// - $S$: symmetric matrix
    fn cartan_decomposition(self) -> Option<(Self, Self)>;
    /// Returns $(K, A, N)$ of the Iwasawa decomposition: $M = KAN$.
    ///
    /// - $K$: orthonormal matrix
    /// - $A$: diagonal matrix
    /// - $N$: upper-half unipotent matrix
    fn iwasawa_decomposition(self) -> Option<(Self, Self, Self)>;
}
