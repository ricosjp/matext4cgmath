use cgmath::*;
use matext4cgmath::*;

#[test]
fn matrix2() {
    const EPS: f64 = 1.0e-10;
    let mat = Matrix2::new(2.0, -4.0, 3.0, 1.0);
    assert!(f64::abs(mat.norm_l1() - 6.0) < EPS);
    assert!(f64::abs(mat.norm_l2() - f64::sqrt(15.0 + f64::sqrt(29.0))) < EPS);
    assert!(f64::abs(mat.norm_linf() - 5.0) < EPS);
}

#[test]
fn matrix3() {
    const EPS: f64 = 1.5e-10;
    #[rustfmt::skip]
    let mat = Matrix3::new(
        1.0, 1.0, 1.0,
        -1.0, 0.0, 1.0,
        1.0, -2.0, 1.0,
    );
    assert!(f64::abs(mat.norm_l1() - 4.0) < EPS);
    assert!(f64::abs(mat.norm_l2() - f64::sqrt(6.0)) < EPS);
    assert!(f64::abs(mat.norm_linf() - 3.0) < EPS);
}
