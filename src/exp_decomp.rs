use crate::*;

impl<F: BaseFloat> Exponential for Matrix2<F> {}
impl<F: BaseFloat> Exponential for Matrix3<F> {}
impl<F: BaseFloat> Exponential for Matrix4<F> {}

fn solve2<F: BaseFloat>(mat: Matrix2<F>) -> Vector2<F> {
    let mat = mat.transpose();
    match mat[0].magnitude2() > mat[1].magnitude2() {
        true => Vector2::new(-mat[0][1], mat[0][0]).normalize(),
        false => Vector2::new(-mat[1][1], mat[1][0]).normalize(),
    }
}

impl<F: BaseFloat> Decomposition for Matrix2<F> {
    fn iwasawa_decomposition(self) -> Option<(Self, Self, Self)> {
        let v0 = self[0];
        let mag0 = dot(v0, v0);
        if mag0 <= F::zero() {
            return None;
        }
        let n0 = dot(self[1], v0) / mag0;
        let v1 = self[1] - v0 * n0;
        let mag1 = dot(v1, v1);
        if mag1 <= F::zero() {
            return None;
        }
        let (a0, a1) = (F::sqrt(mag0), F::sqrt(mag1));
        Some((
            Matrix2::from_cols(v0 / a0, v1 / a1),
            Matrix2::from_diagonal(Vector2::new(a0, a1)),
            Matrix2::new(F::one(), F::zero(), n0, F::one()),
        ))
    }
    fn cartan_decomposition(self) -> Option<(Self, Self)> {
        let a = self.transpose() * self;
        if !a.is_invertible() {
            return None;
        } else if a.is_diagonal() {
            let diag0 = Vector2::new(F::signum(self[0][0]), F::signum(self[1][1]));
            let diag1 = Vector2::new(F::ln(F::abs(self[0][0])), F::ln(F::abs(self[1][1])));
            return Some((Self::from_diagonal(diag0), Self::from_diagonal(diag1)));
        }

        let k = a.eigenvalues();
        let k = [k[0].re, k[1].re];
        let k = Matrix2::from_cols(
            solve2(a - Matrix2::from_value(k[0])),
            solve2(a - Matrix2::from_value(k[1])),
        );
        let diag = k.transpose() * a * k;
        let x = Vector2::new(F::ln(diag[0][0]), F::ln(diag[1][1])) / (F::one() + F::one());
        let x = k * Matrix2::from_diagonal(x) * k.transpose();

        let q = Vector2::new(F::sqrt(diag[0][0]), F::sqrt(diag[1][1]));
        let sym = k * Matrix2::from_diagonal(q) * k.transpose();
        let k = self * sym.invert().unwrap();
        Some((k, x))
    }
}

fn solve3<F: BaseFloat>(mat: Matrix3<F>) -> Vector3<F> {
    let mat = mat.transpose();
    let z0 = mat[1].cross(mat[2]);
    let z1 = mat[2].cross(mat[0]);
    let z2 = mat[0].cross(mat[1]);
    let mag0 = z0.magnitude2();
    let mag1 = z1.magnitude2();
    let mag2 = z2.magnitude2();
    match (mag0 > mag1, mag1 > mag2, mag2 > mag0) {
        (true, _, false) => z0.normalize(),
        (false, true, _) => z1.normalize(),
        _ => z2.normalize(),
    }
}

fn get_normals<F: BaseFloat>(v: Vector3<F>) -> (Vector3<F>, Vector3<F>) {
    let av0 = F::abs(v[0]);
    let av1 = F::abs(v[1]);
    let av2 = F::abs(v[2]);
    match (av0 > av1, av1 > av2, av2 > av0) {
        (true, _, false) => (
            Vector3::new(-v[1], v[0], F::zero()).normalize(),
            Vector3::new(-v[2], F::zero(), v[0]).normalize(),
        ),
        (false, true, _) => (
            Vector3::new(v[1], -v[0], F::zero()).normalize(),
            Vector3::new(F::zero(), -v[2], v[1]).normalize(),
        ),
        _ => (
            Vector3::new(v[2], F::zero(), -v[0]).normalize(),
            Vector3::new(F::zero(), v[2], -v[1]).normalize(),
        ),
    }
}

fn degenerate2_solve3<F: BaseFloat>(mat: Matrix3<F>) -> (Vector3<F>, Vector3<F>) {
    let mat = mat.transpose();
    let mag0 = mat[0].magnitude2();
    let mag1 = mat[1].magnitude2();
    let mag2 = mat[2].magnitude2();
    match (mag0 > mag1, mag1 > mag2, mag2 > mag0) {
        (true, _, false) => get_normals(mat[0]),
        (false, true, _) => get_normals(mat[1]),
        _ => get_normals(mat[0]),
    }
}

impl<F: BaseFloat> Decomposition for Matrix3<F> {
    fn iwasawa_decomposition(self) -> Option<(Self, Self, Self)> {
        let v0 = self[0];
        let mag0 = dot(v0, v0);
        if mag0 <= F::zero() {
            return None;
        }
        let n0 = dot(self[1], v0) / mag0;
        let v1 = self[1] - v0 * n0;
        let mag1 = dot(v1, v1);
        if mag1 <= F::zero() {
            return None;
        }
        let n1 = dot(self[2], v0) / mag0;
        let n2 = dot(self[2], v1) / mag1;
        let v2 = self[2] - v0 * n1 - v1 * n2;
        let mag2 = dot(v2, v2);
        if mag2 <= F::zero() {
            return None;
        }
        let (a0, a1, a2) = (F::sqrt(mag0), F::sqrt(mag1), F::sqrt(mag2));
        #[rustfmt::skip]
        let res = Some((
            Matrix3::from_cols(v0 / a0, v1 / a1, v2 / a2),
            Matrix3::from_diagonal(Vector3::new(a0, a1, a2)),
            Matrix3::new(
                F::one(), F::zero(), F::zero(),
                n0, F::one(), F::zero(),
                n1, n2, F::one(),
            ),
        ));
        res
    }
    fn cartan_decomposition(self) -> Option<(Self, Self)> {
        let a = self.transpose() * self;
        if !a.is_invertible() {
            return None;
        }

        let eps_2 = F::sqrt(F::epsilon());
        let k = a.eigenvalues();
        let k = [k[0].re, k[1].re, k[2].re];
        let k = match (
            F::abs(k[0] - k[1]) < eps_2,
            F::abs(k[1] - k[2]) < eps_2,
            F::abs(k[2] - k[0]) < eps_2,
        ) {
            (false, false, false) => Matrix3::from_cols(
                solve3(a - Matrix3::from_value(k[0])),
                solve3(a - Matrix3::from_value(k[1])),
                solve3(a - Matrix3::from_value(k[2])),
            ),
            (true, false, false) => {
                let (v0, v1) = degenerate2_solve3(
                    a - Matrix3::from_value((k[0] + k[1]) / (F::one() + F::one())),
                );
                Matrix3::from_cols(v0, v1, solve3(a - Matrix3::from_value(k[2])))
            }
            (false, true, false) => {
                let (v0, v1) = degenerate2_solve3(
                    a - Matrix3::from_value((k[1] + k[2]) / (F::one() + F::one())),
                );
                Matrix3::from_cols(v0, v1, solve3(a - Matrix3::from_value(k[0])))
            }
            (false, false, true) => {
                let (v0, v1) = degenerate2_solve3(
                    a - Matrix3::from_value((k[2] + k[0]) / (F::one() + F::one())),
                );
                Matrix3::from_cols(v0, v1, solve3(a - Matrix3::from_value(k[1])))
            }
            _ => {
                let diag0 = Vector3::new(
                    F::signum(self[0][0]),
                    F::signum(self[1][1]),
                    F::signum(self[2][2]),
                );
                let diag1 = Vector3::new(
                    F::ln(F::abs(self[0][0])),
                    F::ln(F::abs(self[1][1])),
                    F::ln(F::abs(self[2][2])),
                );
                return Some((Self::from_diagonal(diag0), Self::from_diagonal(diag1)));
            }
        };
        let diag = k.transpose() * a * k;
        let x = Vector3::new(F::ln(diag[0][0]), F::ln(diag[1][1]), F::ln(diag[2][2]))
            / (F::one() + F::one());
        let x = k * Matrix3::from_diagonal(x) * k.transpose();

        let q = Vector3::new(
            F::sqrt(diag[0][0]),
            F::sqrt(diag[1][1]),
            F::sqrt(diag[2][2]),
        );
        let sym = k * Matrix3::from_diagonal(q) * k.transpose();
        let mut k = self * sym.invert().unwrap();
        k = k.iwasawa_decomposition().unwrap().0;
        Some((k, x))
    }
}
