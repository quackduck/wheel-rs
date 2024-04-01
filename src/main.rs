use crate::network::*;
use rand::Rng;
use rand::prelude::SliceRandom;

mod network;

fn main() {
    let mut n = new_network(vec![1, 16, 16, 1]);
    let samples = gen_samples();
    for i in 0..100000+1 {
        for (x, y) in &samples {
            n.forward(x.to_vec());
            n.backward(y.to_vec())
        }
        if i % 1000 == 0 {
            println!("Iteration: {}", i);
            test_network(&mut n);
        }
    }
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
    for i in ((-1.0/step_size) as i8..(1.0/step_size) as i8).map(|x| x as f64 * step_size)  {
        println!("{} -> {:?}", i, n.forward(vec![i]));
    }
}
