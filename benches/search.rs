use criterion::{black_box, criterion_group, criterion_main, Criterion};
use hnsw::{distance::euclidean::EuclideanDistance, hnsw::Hnsw};
use rand::{thread_rng, Rng};

fn search_benchmark(c: &mut Criterion) {
    let mut rng = thread_rng();
    for _ in 0..5 {
        let mut hnsw = Hnsw::<32, EuclideanDistance>::default();

        for _ in 0..100000 {
            let vec = rng.gen();
            hnsw.insert(vec);
        }

        c.bench_function("nearest_neighbour_search", |b| {
            b.iter(|| {
                let search_vec = rng.gen();
                let res = hnsw.nearest_neighbour_search(black_box(&search_vec));
                black_box(res)
            })
        });
    }
}

criterion_group!(benches, search_benchmark);
criterion_main!(benches);
