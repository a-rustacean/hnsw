use crate::node::Vector;

use super::DistanceFunction;

pub struct EuclideanDistance;

impl<const DIM: usize> DistanceFunction<DIM> for EuclideanDistance {
    fn distance(x: &Vector<DIM>, y: &Vector<DIM>) -> f32 {
        let sum: f32 = x.iter().zip(y.iter()).map(|(x, y)| (x - y).powi(2)).sum();
        sum.sqrt()
    }
}
