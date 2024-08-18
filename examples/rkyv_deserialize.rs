use hnsw::{distance::euclidean::EuclideanDistance, hnsw::Hnsw};

fn main() {
    let serialized = std::fs::read("./hnsw.rkyv").unwrap();
    let hnsw: Hnsw<32, EuclideanDistance, 6> = unsafe { rkyv::from_bytes_unchecked() }.unwrap();

    println!(
        "Search results: {:#?}",
        hnsw.nearest_neighbour_search(&[1.0; 32])
    );
}
