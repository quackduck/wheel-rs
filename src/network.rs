use rand_distr::{num_traits::Float, Distribution, Normal};


#[derive(Debug)]
#[derive(Clone)]
pub struct Layer {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    activations: Vec<f64>,
    zs: Vec<f64>,
}

pub fn new_layer(weights: Vec<Vec<f64>>, biases: Vec<f64>) -> Layer {
    let activations = vec![0.0; biases.len()];
    let zs = vec![0.0; biases.len()];
    // dbg!(activations.len());
    // dbg!(weights.len());
    Layer {
        weights,
        biases,
        activations,
        zs,
    }
}

pub fn new_random_layer(input_size: usize, this_size: usize) -> Layer {
    let weights = random_weights(input_size, this_size);
    let biases = random_biases(this_size);
    new_layer(weights, biases)
}

fn random_biases(n: usize) -> Vec<f64> {
    vec![0.0; n]
    // vec![-0.24; n]
    // let mut rng = rand::thread_rng();
    // let variance = 1.0 / n as f64;
    // let normal = Normal::new(0.0, variance.sqrt()).unwrap();
    // let mut biases = Vec::with_capacity(n);
    // for _ in 0..n {
    //     biases.push(normal.sample(&mut rng));
    // }
    // biases
}

pub fn print_network_stats(n: &Network) {
    for (i, l) in n.layers.iter().enumerate() {
        if i == 0 {
            continue;
        }
        let weights = flatten(&l.weights);
        let biases = &l.biases;
        println!("layer {} ", i);
        print!("    weights: ");
        print_stats(&weights);
        print!("    biases: ");
        print_stats(biases);
    }
}

fn flatten(v: &Vec<Vec<f64>>) -> Vec<f64> {
    let mut res = Vec::new();
    for row in v {
        res.extend(row);
    }
    res
}

fn print_stats(v: &Vec<f64>) {
    let min = v.iter().fold(f64::INFINITY, |a, &b| a.min(b));
    let max = v.iter().fold(f64::NEG_INFINITY, |a, &b| a.max(b));
    let sum = v.iter().sum::<f64>();
    let avg = sum / v.len() as f64;
    println!("min: {:.2}, max: {:.2}, avg: {:.2}", min, max, avg);
}

fn random_weights(input_size: usize, this_size: usize) -> Vec<Vec<f64>> {
    let mut rng = rand::thread_rng();
    let variance = 2.0 / (input_size + this_size) as f64;
    let normal = Normal::new(0.0, variance.sqrt()).unwrap();
    let mut weights = Vec::with_capacity(this_size);
    for _ in 0..this_size {
        let mut row = Vec::with_capacity(input_size);
        for _ in 0..input_size {
            row.push(normal.sample(&mut rng));
        }
        weights.push(row);
    }
    weights
}

fn activate(x: f64) -> f64 {
    x.tanh()
}

fn activate_prime(x: f64) -> f64 {
    let a = activate(x);
    1.0 - a * a // + 0.1 // add a small value to make derivatives not too small
}

fn cost(err: f64) -> f64 {
    err * err
}

fn cost_prime(err: f64) -> f64 {
    // 2.0 * err.powi(3)
    // 2.0 * err
    err
}

impl Layer {
    pub fn forward(&mut self, inputs: Vec<f64>) {
        for i in 0..self.biases.len() {
            let z = self.biases[i] + self.weights[i].iter().zip(inputs.iter()).map(|(w, x)| w * x).sum::<f64>();
            self.zs[i] = z;
            self.activations[i] = activate(z);
        }
    }
}

#[derive(Debug)]
#[derive(Clone)]
pub struct Network {
    layers: Vec<Layer>,
    all_last_weight_updates: Vec<Vec<Vec<f64>>>, // for each layer, for each neuron, for each weight
    all_last_bias_updates: Vec<Vec<f64>>, // for each layer, for each neuron
    // cost: f64,
}

pub fn new_network(sizes: Vec<usize>) -> Network {
    let mut layers = Vec::with_capacity(sizes.len() - 1);
    layers.push(new_random_layer(0, sizes[0]));
    for i in 0..sizes.len() - 1 {
        layers.push(new_random_layer(sizes[i], sizes[i + 1]));
    }
    let mut all_last_weight_updates = Vec::with_capacity(layers.len());
    let mut all_last_bias_updates = Vec::with_capacity(layers.len());
    for i in 0..layers.len() {
        let layer = &layers[i];
        let mut lastWeightUpdates = Vec::with_capacity(layer.weights.len());
        for j in 0..layer.weights.len() {
            lastWeightUpdates.push(vec![0.0; layer.weights[j].len()]);
        }
        all_last_weight_updates.push(lastWeightUpdates);
        all_last_bias_updates.push(vec![0.0; layer.biases.len()]);
    }
    Network {
        layers,
        all_last_weight_updates,
        all_last_bias_updates,
        // cost: 0.0,
    }
}

impl Network {
    pub fn forward(&mut self, inputs: Vec<f64>) -> Vec<f64> {
        self.layers[0].activations = inputs;
        for i in 1..self.layers.len() {
            let prev_activations = self.layers[i - 1].activations.clone();
            self.layers[i].forward(prev_activations);
        }
        self.layers.last().unwrap().activations.clone()
    }

    pub fn backward(&mut self, wanted: Vec<f64>) {
        // self.cost = 0.0;
        // let mut idx = self.layers.len() - 1;

        // dbg!(&self.layers);

        let (mut last_layer, rest) = self.layers.split_last_mut().unwrap();
        // last_layer.activations.iter().zip(wanted.iter()).for_each(|(a, w)| {
        //     self.cost += cost(a-w);
        // });

        let rate = 0.001;

        let mut deltas = vec![0.0; last_layer.weights.len()];
        for i in 0..last_layer.activations.len() {
            deltas[i] = 2.0 * cost_prime(last_layer.activations[i] - wanted[i]) * activate_prime(last_layer.zs[i]);
        }
        let second_last_activs = &rest.last_mut().unwrap().activations;
        update_layer_with_deltas(&mut last_layer, &second_last_activs, &deltas, rate, 
            &mut self.all_last_weight_updates.last_mut().unwrap(), &mut self.all_last_bias_updates.last_mut().unwrap());
        // update_layer_with_deltas(&mut last_layer, &second_last_activs, &deltas, rate);

        for i in (1..rest.len()).rev() { // we already handled the last one, and no need to update the input layer
            
            let layer = &self.layers[i];
            let mut next_deltas = vec![0.0; layer.weights.len()];
            for j in 0..layer.weights.len() {
                let mut sum = 0.0;
                for k in 0..self.layers[i + 1].weights.len() {
                    sum += self.layers[i + 1].weights[k][j] * deltas[k];
                    // sum = self.layers[i + 1].weights[k][j].mul_add( deltas[k], sum);
                }
                next_deltas[j] = sum * activate_prime(layer.zs[j]);
            }

            // let prev_activs = self.layers[i - 1].activations.clone();
            let (before, this) = self.layers.split_at_mut(i);
            // update_layer_with_deltas(&mut this[0], &before.last().unwrap().activations, &next_deltas, rate);
            update_layer_with_deltas(&mut this[0], &before.last().unwrap().activations, &next_deltas, rate, 
                &mut self.all_last_weight_updates[i], &mut self.all_last_bias_updates[i]);
            deltas = next_deltas;
        }
    }
}

fn err_to_delcdelw(delta: f64, prevactiv: f64) -> f64 {
    delta * prevactiv
}

fn err_to_delcdelb(delta: f64) -> f64 {
    delta
}

fn update_layer_with_deltas(layer: &mut Layer, prev_activations: &Vec<f64>, deltas: &Vec<f64>, learning_rate: f64, 
    weight_updates: &mut Vec<Vec<f64>>, bias_updates: &mut Vec<f64>) {

    let momentum = 0.5;

    for i in 0..layer.weights.len() { // for every neuron
        for j in 0..layer.weights[i].len() { // for every weight in the neuron
            weight_updates[i][j] = (learning_rate * err_to_delcdelw(deltas[i], prev_activations[j]) +
                momentum * weight_updates[i][j]); // /(1.0 + momentum);
            layer.weights[i][j] -= weight_updates[i][j];
        }
    }

    for i in 0..layer.biases.len() { // for every neuron
        bias_updates[i] = (learning_rate * err_to_delcdelb(deltas[i]) +
            momentum * bias_updates[i]); // / (1.0 + momentum);
        layer.biases[i] -= bias_updates[i];
    }
}
