#[macro_use(array)]
extern crate ndarray;
extern crate gnuplot;

use gnuplot::{Figure, Caption, Color};
use ndarray::{Array2, Array};

fn gradient_descent(f: fn(&Array2<f64>, &Array2<f64>) -> f64,
                    df: fn(&Array2<f64>, &Array2<f64>) -> Array2<f64>,
                    mut x: Array2<f64>,
                    a: &Array2<f64>,
                    steps: usize,
                    lr: f64) -> (Array2<f64>, Vec<f64>) {
    let mut loss = Vec::<f64>::new();
    for _ in 0..steps {
        let dx = df(a, &x);
        x = x - dx * lr;
        loss.push(f(a, &x));
    }
    (x, loss)
}

fn rmsprop(f: fn(&Array2<f64>, &Array2<f64>) -> f64,
           df: fn(&Array2<f64>, &Array2<f64>) -> Array2<f64>,
           mut x: Array2<f64>,
           a: &Array2<f64>,
           steps: usize,
           lr: f64,
           decay: f64,
           eps: f64) -> (Array2<f64>, Vec<f64>) {
    let mut loss = Vec::<f64>::new();
    let mut dx_mean_sqr = Array2::<f64>::zeros((x.rows(), x.cols()));
    for _ in 0..steps {
        let mut dx = df(a, &x);
        dx.iter_mut().for_each(|x| { *x = x.powi(2) * (1. - decay); });
        dx_mean_sqr = dx_mean_sqr * decay + &dx;
        let mut dx_mean_sqr_x = dx_mean_sqr.clone();
        dx_mean_sqr_x.iter_mut().for_each(|x| { *x = x.sqrt() + eps; });
        let dx = df(a, &x);
        x = x - dx * lr / &dx_mean_sqr_x;
        loss.push(f(a, &x));
    }
    (x, loss)
}

fn rmsprop_momentum(f: fn(&Array2<f64>, &Array2<f64>) -> f64,
                    df: fn(&Array2<f64>, &Array2<f64>) -> Array2<f64>,
                    mut x: Array2<f64>,
                    a: &Array2<f64>,
                    steps: usize,
                    lr: f64,
                    decay: f64,
                    eps: f64,
                    mu: f64) -> (Array2<f64>, Vec<f64>) {
    let mut loss = Vec::<f64>::new();
    let mut dx_mean_sqr = Array2::<f64>::zeros((x.rows(), x.cols()));
    let mut momentum = Array2::<f64>::zeros((x.rows(), x.cols()));
    for _ in 0..steps {
        let mut dx = df(a, &x);
        dx.iter_mut().for_each(|x| { *x = x.powi(2) * (1. - decay); });
        dx_mean_sqr = dx_mean_sqr * decay + &dx;
        let mut dx_mean_sqr_x = dx_mean_sqr.clone();
        dx_mean_sqr_x.iter_mut().for_each(|x| { *x = x.sqrt() + eps; });
        let dx = df(a, &x);
        momentum = &momentum * mu + dx * lr / &dx_mean_sqr_x;
        x = x - &momentum;
        loss.push(f(a, &x));
    }
    (x, loss)
}

fn f(a: &Array2<f64>, x: &Array2<f64>) -> f64 {
    let x_len = x.rows();
    let residual = a.dot(x) - Array::eye(x_len);
    residual.iter().map(|&x| x.powi(2)).sum()
}

fn df(a: &Array2<f64>, x: &Array2<f64>) -> Array2<f64> {
    a.t().dot(&(a.dot(x) - &Array2::eye(x.rows()))) * 2.
}

fn main() {
    let a = array![
    [2., 5., 1., 4., 6.],
    [3., 5., 0., 0., 0.],
    [1., 1., 0., 3., 8.],
    [6., 6., 2., 2., 1.],
    [8., 3., 5., 1., 4.],
 ];
    let (x1, loss1) = gradient_descent(f, df, &a * 0., &a, 300, 0.001);
    let x1_x = (0..loss1.len()).collect::<Vec<usize>>();
    println!("{} {}", a.dot(&x1), loss1[loss1.len()-1]);

    let (x2, loss2) = rmsprop(f, df, &a * 0., &a, 300, 0.001, 0.9, 1e-8);
    let x2_x = (0..loss2.len()).collect::<Vec<usize>>();
    println!("{} {}", a.dot(&x2), loss2[loss2.len()-1]);

    let (x3, loss3) = rmsprop_momentum(f, df, &a * 0., &a, 300, 0.001, 0.9, 1e-8, 0.9);
    let x3_x = (0..loss3.len()).collect::<Vec<usize>>();
    println!("{} {}", a.dot(&x3), loss3[loss3.len()-1]);
    let mut fg = Figure::new();

    fg.axes2d()
        .lines(&x1_x, &loss1, &[Caption("gradient descent"), Color("black")])
        .lines(&x2_x, &loss2, &[Caption("rmsprop"), Color("green")])
        .lines(&x3_x, &loss3, &[Caption("rmsprop+momentum"), Color("red")]);
    fg.show();
}
