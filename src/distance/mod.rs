pub mod euclidean;

use crate::node::Vector;

pub trait DistanceFunction<const DIM: usize> {
    fn distance(x: &Vector<DIM>, y: &Vector<DIM>) -> f32;
}
