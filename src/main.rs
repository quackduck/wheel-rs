use crate::network::*;
use crate::mnist::*;
// use rand::Rng;
use rand::prelude::SliceRandom;
use std::time::Instant;

mod network;
mod mnist;

fn main() {
    let mut n = new_network(vec![784, 64, 64, 10]);
    
    let m = new_mnist();
    
    // start timer
    let now = Instant::now();

    for i in 0..15+1 {
        let samples = mnist_task(&m, 0); // returns all train data
        // if i % 60 == 0 {
        println!("Epoch: {}", i);
        mnist_test(&m, &mut n);
        // }
        
        // dbg!(samples.len());
        for (x, y) in &samples {
            n.forward(x.to_vec());
            n.backward(y.to_vec())
        }
    }
    println!("Time elapsed: {:?}", now.elapsed());
}

fn simple_task() -> Vec<(Vec<f64>, Vec<f64>)> {
    let mut samples = Vec::with_capacity(300);
    for i in 0..100 {
        samples.push((vec![0.0], vec![1.0]));
        samples.push((vec![-1.0], vec![-1.0]));
        samples.push((vec![1.0], vec![0.0]));
    }
    samples.shuffle(&mut rand::thread_rng());
    samples
}

fn gen_samples() -> Vec<(Vec<f64>, Vec<f64>)> {
    // let mut rng = rand::thread_rng();
    // let mut samples = Vec::with_capacity(n);
    // for _ in 0..n {
    //     let x = rng.gen_range(-1.0..1.0);
    //     let y = x *x;
    //     samples.push((vec![x], vec![y]));
    // }
    // samples

    let mut samples = Vec::with_capacity(100);
    let step_size = 0.1;
    for i in ((-1.0/step_size) as i8..(1.0/step_size) as i8 + 1).map(|x| x as f64 * step_size)  {
        samples.push((vec![i], vec![i * i]));
    }
    samples.shuffle(&mut rand::thread_rng());
    samples
}

fn test_network(n: &mut Network) {
    let step_size = 0.1;
    for i in ((-1.0/step_size) as i8..(1.0/step_size) as i8 + 1).map(|x| x as f64 * step_size)  {
        println!("{} -> {:?}", i, n.forward(vec![i]));
    }
}
