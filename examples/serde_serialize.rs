use hnsw::{distance::euclidean::EuclideanDistance, hnsw::Hnsw};
use rand::{rng, Rng};

fn main() {
    let mut hnsw = Hnsw::<32, EuclideanDistance, 6>::default();
    let mut rng = rng();

    for _ in 0..10000 {
        let vec = rng.random();
        hnsw.insert(vec);
    }

    let serialized = serde_json::to_string_pretty(&hnsw).unwrap();

    std::fs::write("./hnsw.json", serialized).unwrap();
}
