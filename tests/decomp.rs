use cgmath::*;
use matext4cgmath::*;

#[test]
fn matrix2() {
    (0..10000).for_each(|_i| {
        let mat = Matrix2::new(
            10.0 * rand::random::<f64>() - 5.0,
            10.0 * rand::random::<f64>() - 5.0,
            10.0 * rand::random::<f64>() - 5.0,
            10.0 * rand::random::<f64>() - 5.0,
        );
        if mat.is_invertible() {
            let (k, a, n) = mat.iwasawa_decomposition().unwrap();
            let res = mat - k * a * n;
            assert!(res.norm_l1() < 1.0e-8, "{_i} {k:?}\n{a:?}\n{n:?}\n{res:?}");
        }
    });
}

#[test]
fn matrix3() {
    (0..10000).for_each(|_i| {
        let mat = Matrix3::new(
            10.0 * rand::random::<f64>() - 5.0,
            10.0 * rand::random::<f64>() - 5.0,
            10.0 * rand::random::<f64>() - 5.0,
            10.0 * rand::random::<f64>() - 5.0,
            10.0 * rand::random::<f64>() - 5.0,
            10.0 * rand::random::<f64>() - 5.0,
            10.0 * rand::random::<f64>() - 5.0,
            10.0 * rand::random::<f64>() - 5.0,
            10.0 * rand::random::<f64>() - 5.0,
        );
        if mat.is_invertible() {
            let (k, a, n) = mat.iwasawa_decomposition().unwrap();
            let res = mat - k * a * n;
            assert!(res.norm_l1() < 1.0e-8, "{_i} {k:?}\n{a:?}\n{n:?}\n{res:?}");
        }
    });
}
