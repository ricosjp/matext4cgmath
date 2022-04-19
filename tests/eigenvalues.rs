use cgmath::*;
use matext4cgmath::*;
use std::f64::consts::PI;

#[test]
fn matrix2() {
    const EPS: f64 = 1.0e-10;

    // real number case
    let mat = Matrix2::new(-4.0, -2.0, 5.0, 3.0);
    let eigens = mat.eigenvalues();
    assert!((eigens[0] + 2.0).norm() < EPS);
    assert!((eigens[1] - 1.0).norm() < EPS);

    // pure imaginary
    let mat = Matrix2::new(0.0, 1.0, -1.0, 0.0);
    let eigens = mat.eigenvalues();
    assert!((eigens[0] * eigens[0] + 1.0).norm() < EPS);
    assert!((eigens[1] * eigens[1] + 1.0).norm() < EPS);
    assert!((eigens[0] - eigens[1]).norm() > 1.0);

    // complex
    let mat = Matrix2::new(1.0, 1.0, -1.0, 1.0);
    let eigens = mat.eigenvalues();
    assert!(((eigens[0] - 1.0) * (eigens[0] - 1.0) + 1.0).norm() < EPS);
    assert!(((eigens[1] - 1.0) * (eigens[1] - 1.0) + 1.0).norm() < EPS);
    assert!((eigens[0] - eigens[1]).norm() > 1.0);
}

fn random_unit3() -> Vector3<f64> {
    let u = Vector2::new(rand::random::<f64>(), rand::random::<f64>());
    let theta = 2.0 * std::f64::consts::PI * u[0];
    let z = 2.0 * u[1] - 1.0;
    let r = f64::sqrt(1.0 - z * z);
    Vector3::new(r * f64::cos(theta), r * f64::sin(theta), z)
}

fn random_vector3() -> Vector3<f64> {
    Vector3::new(
        10.0 * rand::random::<f64>() - 5.0,
        10.0 * rand::random::<f64>() - 5.0,
        10.0 * rand::random::<f64>() - 5.0,
    )
}

#[test]
fn matrix3() {
    const EPS: f64 = 1.0e-8;

    // random
    (0..1000).for_each(|_i| {
        let diag = random_vector3();
        let p = Matrix3::from_axis_angle(random_unit3(), Rad(2.0 * PI * rand::random::<f64>()));
        let ng = random_vector3();
        #[rustfmt::skip]
        let nilp = Matrix3::new(
            1.0, 0.0, 0.0,
            ng.x, 1.0, 0.0,
            ng.y, ng.z, 1.0,
        );
        let mat = p * nilp * Matrix3::from_diagonal(diag) * (p * nilp).invert().unwrap();
        let eigens = mat.eigenvalues();
        [diag.x, diag.y, diag.z].iter().for_each(|e| {
            let any = eigens.iter().any(|x| (e - x).norm() < EPS);
            assert!(any, "{mat:?} {diag:?} {eigens:?}");
        });
    });
}

#[test]
fn matrix4() {
    const EPS: f64 = 1.0e-10;

    #[rustfmt::skip]
    let mat = Matrix4::new(
        67.0, -6.0, -21.0, -55.0,
        -654.0, 32.0, 138.0, 510.0,
        507.0, -22.0, -101.0, -395.0,
        -2.0, -4.0, -10.0, -2.0,
    );
    let mut eigens = mat.eigenvalues();
    eigens.sort_by(|x, y| x.re.partial_cmp(&y.re).unwrap());
    [-12.0, -4.0, 4.0, 8.0].iter().copied().for_each(|x| {
        let any = eigens.iter().any(|e| (e - x).norm() < EPS);
        assert!(any, "{eigens:?}");
    });
}
