use hnsw::{distance::euclidean::EuclideanDistance, hnsw::Hnsw};
use rand::{thread_rng, Rng};

fn main() {
    let mut hnsw = Hnsw::<32, EuclideanDistance, 6>::default();
    let mut rng = thread_rng();

    for _ in 0..10000 {
        let vec = rng.gen();
        hnsw.insert(vec);
    }

    let serialized = rkyv::to_bytes(&hnsw).unwrap();

    std::fs::write("./hnsw.rkyv", serialized).unwrap();
}
