use mnist::*;
use crate::network::*;
use rand::Rng;

pub struct Mnist {
    trn_img: Vec<Vec<u8>>,
    trn_lbl: Vec<u8>,
    tst_img: Vec<Vec<u8>>,
    tst_lbl: Vec<u8>,
}

pub fn new_mnist() -> Mnist {
    let m = MnistBuilder::new()
        .label_format_digit()
        // .label_format_one_hot()
        .training_set_length(60_000)
        .validation_set_length(0)
        .test_set_length(10_000)
        .finalize();
    let mut new_trn_img = Vec::with_capacity(m.trn_lbl.len());
    dbg!(m.trn_img.len());
    dbg!(m.trn_lbl.len());
    // correct the shape of the data
    for i in 0..m.trn_lbl.len() {
        let mut img = Vec::with_capacity(28 * 28);
        let idx = i * 28 * 28;
        img.extend_from_slice(&m.trn_img[idx..idx + 28 * 28]);
        new_trn_img.push(img);
    }
    let mut new_tst_img = Vec::with_capacity(m.tst_lbl.len());
    for i in 0..m.tst_lbl.len() {
        let mut img = Vec::with_capacity(28 * 28);
        let idx = i * 28 * 28;
        img.extend_from_slice(&m.tst_img[idx..idx + 28 * 28]);
        new_tst_img.push(img);
    }
    Mnist {
        trn_img: new_trn_img,
        trn_lbl: m.trn_lbl,
        tst_img: new_tst_img,
        tst_lbl: m.tst_lbl,
    }
}

pub fn mnist_task(mnist: &Mnist, iteration: usize) -> Vec<(Vec<f64>, Vec<f64>)> {
    let batch = 60000;
    let mut samples = Vec::with_capacity(batch);
    // let skipby = (batch * iteration) % (mnist.trn_lbl.len() - batch);
    // for i in skipby..skipby + batch {
    for i in 0..batch {
        let img = &mnist.trn_img[i];
        let lbl = &mnist.trn_lbl[i];
        let img = img.iter().map(|x| *x as f64 / 255.0).collect();
        let mut lbl_vec = vec![-1.0; 10];
        lbl_vec[*lbl as usize] = 1.0;
        samples.push((img, lbl_vec));
    }
    samples
}

pub fn mnist_test(mnist: &Mnist, n: &mut Network) {
    println!("Testing...");
    let mut correct = 0;
    let batch = 10000;
    // let skipby = rand::thread_rng().gen_range(0..mnist.tst_lbl.len() - batch);
    // for i in skipby..skipby + batch {
    for i in 0..batch {
        let img = &mnist.tst_img[i];
        let lbl = &mnist.tst_lbl[i];
        let img = img.iter().map(|x| *x as f64 / 255.0).collect();
        let lbl = *lbl as usize;
        let res = n.forward(img);
        let guess = res.iter().enumerate().max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
        if guess == lbl {
            correct += 1;
        }
    }
    println!("Correct: {}/{} = {:.1}", correct, batch, 100.0 * correct as f64 / batch as f64);
}