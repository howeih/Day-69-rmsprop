Day 69: rmsprop
<br>
Rmsprop is a gradient-based optimization technique proposed by Geoffrey Hinton at his Neural Networks Coursera course.
<br>
The concept of neural networks has been known for decades, but researchers have been failing to train any kind of slightly complex network. While there were more reasons to that, the one that was very hard to address was a gradient magnitude.
Gradients of very complex functions like neural networks have a tendency to either vanish or explode as the energy is propagated through the function. And the effect has a cumulative nature — the more complex the function is, the worse the problem becomes.
Rmsprop is a very clever way to deal with the problem. It uses a moving average of squared gradients to normalize the gradient itself. That has an effect of balancing the step size — decrease the step for large gradient to avoid exploding, and increase the step for small gradient to avoid vanishing.
<br>
I have implemented three versions of gradient-based techniques.
<br>

* gradient descent
* rmsprop
* rmsprop with momentum

And I will use them to find an inversion of matrix A. The loss function will be a squared difference between AX and unit matrix I with the corresponding derivative.
<br>
![Alt text](1.png?raw=true)
<br>
Here is the plot of loss function for each method with respect to the number of steps.
<br>
![Alt text](rmsprop.png?raw=true)
<br>
run:

```
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

```
result:

```
[[0.7887994493417734, -0.013393893555995295, 0.17895029874234003, 0.1860322819946345, -0.08252066058896279],
 [-0.013393893555995073, 0.8016511119690202, 0.004871810807576427, 0.196724184030231, -0.06795410134228565],
 [0.17895029874233992, 0.004871810807576593, 0.8481629460866849, -0.15156070882280903, 0.06786291249848203],
 [0.18603228199463406, 0.19672418403023118, -0.15156070882280898, 0.6629829416446524, 0.13141483842105545],
 [-0.08252066058896285, -0.0679541013422858, 0.06786291249848186, 0.1314148384210555, 0.9478388023633267]] 0.5469198476714345
[[0.8572675164146513, -0.04670293759042499, 0.11918896146998148, 0.10279434569870383, -0.057223557076203524],
 [-0.031212866594801825, 0.8196098847675589, 0.032212342439340624, 0.18670761476540826, -0.03300768590279769],
 [0.13273852589950055, 0.031135390346078595, 0.914862342938974, -0.07787056603790266, 0.03137453240591359],
 [0.16616699317223824, 0.19706827471932425, -0.11018473586575382, 0.7506338954280991, 0.05586737128199315],
 [-0.062364809033979104, -0.0860506113588822, 0.05588607690670416, 0.10304326149228404, 0.9676442214348648]] 0.32396300469758166
[[0.9933824394389017, 0.00746286997341894, 0.00029401285468333427, -0.007514708113346424, 0.0008927172849861886],
 [-0.002940889970965088, 1.0033169214838698, 0.00013111702938017888, -0.003340346036328823, 0.00039652752724395057],
 [-0.004780000804944451, 0.00538997042994116, 1.0002119344452312, -0.005426737237507417, 0.0006451591142516699],
 [-0.00624984048734592, 0.007049001843444175, 0.00027802333531125323, 0.9929024218709792, 0.0008431118255463194],
 [-0.007720096714362426, 0.008707177079675943, 0.0003441240953629965, -0.008767833081697485, 1.001040834715449]] 0.0006230388777265792

```
