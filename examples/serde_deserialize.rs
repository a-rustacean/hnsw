use hnsw::{distance::euclidean::EuclideanDistance, hnsw::Hnsw};

fn main() {
    let serialized = std::fs::read_to_string("./hnsw.json").unwrap();
    let hnsw: Hnsw<32, EuclideanDistance, 6> = serde_json::from_str(&serialized).unwrap();

    println!(
        "Search results: {:#?}",
        hnsw.nearest_neighbour_search(&[1.0; 32])
    );
}
