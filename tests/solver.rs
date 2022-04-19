use matext4cgmath::solver;
use num_complex::Complex;

#[test]
fn solve_quadratic_test() {
    const EPS: f64 = 1.0e-10;
    // example
    let res = solver::solve_quadratic(-2.0, 1.0);
    assert!(Complex::norm(res[0] - 1.0) < EPS);
    assert!(Complex::norm(res[1] - 1.0) < EPS);

    // random
    (0..10000).for_each(|_| {
        let a = 100.0 * rand::random::<f64>() - 50.0;
        let b = 100.0 * rand::random::<f64>() - 50.0;
        let vec = solver::solve_quadratic(a, b);
        vec.into_iter().for_each(|t| {
            let f = t * t + a * t + b;
            let g = f64::max((2.0 * t + a).norm(), 1.0);
            assert!(Complex::norm(f) < EPS * g, "{a} {b} {t} {f}");
        });
    })
}

#[test]
fn pre_solve_cubic_test() {
    const EPS: f64 = 1.49e-8; // sqrt EPSILON
                              // random
    (0..10000).for_each(|i| {
        let p = 100.0 * rand::random::<f64>() - 50.0;
        let q = 100.0 * rand::random::<f64>() - 50.0;
        let vec = solver::pre_solve_cubic(p, q);
        vec.into_iter().for_each(|t| {
            let f = t * t * t + p * t + q;
            let g = f64::max((3.0 * t * t + p).norm(), 1.0);
            assert!(Complex::norm(f) < EPS * g, "{i} {p} {q} {vec:?} {t} {f}");
        });
    })
}

#[test]
fn solve_cubic_test() {
    const EPS: f64 = 1.49e-8; // sqrt EPSILON
                              // example 0
    let mut res = solver::solve_cubic(-1.5, -11.5, 6.0);
    res.sort_by(|x, y| x.re.partial_cmp(&y.re).unwrap());
    let ans = [Complex::from(-3.0), Complex::from(0.5), Complex::from(4.0)];
    res.iter().zip(ans).for_each(|(x, y)| {
        assert!(Complex::norm(x - y) < EPS);
    });

    // example 1
    let res = solver::solve_cubic(-6.0, 12.0, -8.0);
    res.iter().for_each(|x| {
        assert!(Complex::norm(x - 2.0) < EPS);
    });

    // random
    (0..10000).for_each(|_| {
        let a = 100.0 * rand::random::<f64>() - 50.0;
        let b = 100.0 * rand::random::<f64>() - 50.0;
        let c = 100.0 * rand::random::<f64>() - 50.0;
        let vec = solver::solve_cubic(a, b, c);
        vec.into_iter().for_each(|t| {
            let f = t * t * t + a * t * t + b * t + c;
            let g = f64::max((3.0 * t * t + 2.0 * a * t + b).norm(), 1.0);
            assert!(Complex::norm(f) < EPS * g, "{a} {b} {c} {vec:?} {t} {f}");
        });
    });
}

#[test]
fn pre_solve_quartic_test() {
    const EPS: f64 = 1.49e-8; // sqrt EPSILON
                              // random
    (0..10000).for_each(|i| {
        let p = 100.0 * rand::random::<f64>() - 50.0;
        let q = 100.0 * rand::random::<f64>() - 50.0;
        let r = 100.0 * rand::random::<f64>() - 50.0;
        let vec = solver::pre_solve_quartic(p, q, r);
        vec.into_iter().for_each(|t| {
            let f = t * t * t * t + p * t * t + q * t + r;
            let g = f64::max((4.0 * t * t * t + 2.0 * p * t + q).norm(), 1.0);
            assert!(
                Complex::norm(f) < EPS * g,
                "{i} {p} {q} {r} {vec:?} {t} {f}"
            );
        });
    });
}

#[test]
fn solve_quartic_test() {
    const EPS: f64 = 1.49e-8; // sqrt EPSILON
                              // random
    (0..10000).for_each(|i| {
        let a = 100.0 * rand::random::<f64>() - 50.0;
        let b = 100.0 * rand::random::<f64>() - 50.0;
        let c = 100.0 * rand::random::<f64>() - 50.0;
        let d = 100.0 * rand::random::<f64>() - 50.0;
        let vec = solver::solve_quartic(a, b, c, d);
        vec.into_iter().for_each(|t| {
            let f = t * t * t * t + a * t * t * t + b * t * t + c * t + d;
            let g = f64::max(
                (4.0 * t * t * t + 3.0 * a * t * t + 2.0 * b * t + c).norm(),
                1.0,
            );
            assert!(
                Complex::norm(f) < EPS * g,
                "{i} {a} {b} {c} {d} {vec:?} {t} {f}"
            );
        });
    });
}
