use crate::{
    distance::DistanceFunction,
    level::Level,
    node::{GraphNode, GraphNodeIdx, Node, NodeIdx, Vector},
    rand::LCGRng,
};
use arrayvec::ArrayVec;
use index_vec::IndexVec;
use std::{collections::HashSet, fmt, mem::MaybeUninit};

#[derive(Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
pub struct Hnsw<
    const DIM: usize,
    DF: DistanceFunction<DIM>,
    const LEVEL_COUNT: usize = 5,
    const MAX_NEIGHBOURS: usize = 20,
> {
    pub nodes: IndexVec<NodeIdx, Node<DIM>>,
    #[cfg_attr(feature = "serde", serde(with = "serde_big_array::BigArray"))]
    pub levels: [Level<DIM, DF, MAX_NEIGHBOURS>; LEVEL_COUNT],
    pub rng: LCGRng,
    #[cfg_attr(feature = "serde", serde(with = "serde_big_array::BigArray"))]
    pub level_probabilities: [f64; LEVEL_COUNT],
}

impl<
        const DIM: usize,
        DF: DistanceFunction<DIM>,
        const LEVEL_COUNT: usize,
        const MAX_NEIGHBOURS: usize,
    > fmt::Debug for Hnsw<DIM, DF, LEVEL_COUNT, MAX_NEIGHBOURS>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Hnsw")
            .field("nodes", &self.nodes)
            .field("levels", &self.levels)
            .field("rng", &self.rng)
            .field("level_probabilities", &self.level_probabilities)
            .finish()
    }
}

impl<
        const DIM: usize,
        DF: DistanceFunction<DIM>,
        const LEVEL_COUNT: usize,
        const MAX_NEIGHBOURS: usize,
    > Default for Hnsw<DIM, DF, LEVEL_COUNT, MAX_NEIGHBOURS>
{
    fn default() -> Self {
        Self::new(42, 0.5)
    }
}

impl<
        const DIM: usize,
        DF: DistanceFunction<DIM>,
        const LEVEL_COUNT: usize,
        const MAX_NEIGHBOURS: usize,
    > Hnsw<DIM, DF, LEVEL_COUNT, MAX_NEIGHBOURS>
{
    pub fn new(seed: u64, lambda: f64) -> Self {
        let mut levels: [MaybeUninit<Level<DIM, DF, MAX_NEIGHBOURS>>; LEVEL_COUNT] =
            unsafe { MaybeUninit::uninit().assume_init() };
        for level in &mut levels {
            level.write(Level::<DIM, DF, MAX_NEIGHBOURS>::default());
        }
        Self {
            nodes: IndexVec::new(),
            levels: unsafe { std::mem::transmute_copy(&levels) },
            rng: LCGRng::new(seed),
            level_probabilities: generate_probabilities(lambda),
        }
    }

    fn gen_level(&mut self) -> usize {
        let random_value: f64 = self.rng.next_f64(); // Get a random number between 0.0 and 1.0

        // Find the level based on the cumulative probability distribution
        let mut cumulative_prob = 0.0;
        for (level, &prob) in self.level_probabilities.iter().enumerate() {
            cumulative_prob += prob;
            if random_value <= cumulative_prob {
                return level;
            }
        }

        // Fallback in case of precision issues
        LEVEL_COUNT - 1
    }

    pub fn insert(&mut self, vec: Vector<DIM>) {
        let node = Node {
            idx: self.nodes.len_idx(),
            vec,
        };
        let node_idx = self.nodes.push(node);
        self.index_node(node_idx);
    }

    pub fn index_node(&mut self, node_idx: NodeIdx) {
        let level_count = self.gen_level();
        let vec = &self.nodes[node_idx].vec;

        let mut child_graph_node_idx = None;

        for level_idx in 0..level_count {
            let graph_node_idx = {
                let level = &mut self.levels[level_idx];
                let mut neighbour_list = level.nearest_neighbour_search(vec, &self.nodes);
                let mut seen = HashSet::new();
                neighbour_list.retain(|(id, _)| seen.insert(*id));
                neighbour_list.truncate(MAX_NEIGHBOURS);
                let neighbours = ArrayVec::from_iter(neighbour_list.clone());
                let graph_node_idx = level.graph.push(GraphNode {
                    node_idx,
                    neighbours,
                    parent: None,
                    child: child_graph_node_idx,
                });
                level.make_edges_double_sided(graph_node_idx, neighbour_list);
                graph_node_idx
            };
            if let Some(child_graph_node_idx) = child_graph_node_idx {
                let prev_level = &mut self.levels[level_idx - 1];
                prev_level.graph[child_graph_node_idx].parent = Some(graph_node_idx);
            }
            child_graph_node_idx = Some(graph_node_idx);
        }
    }

    pub fn nearest_neighbour_search(&self, query: &Vector<DIM>) -> Vec<(&Node<DIM>, f32)> {
        if self.nodes.is_empty() {
            return Vec::new();
        }

        let mut start_graph_node_idx = GraphNodeIdx::new(0);

        let mut level_idx = LEVEL_COUNT - 1;

        loop {
            let level = &self.levels[level_idx];

            if level.graph.is_empty() {
                level_idx -= 1;
                continue;
            }

            let closest_neighbours = level.nearest_neighbour_search_starting_from(
                query,
                &self.nodes,
                start_graph_node_idx,
            );

            if level_idx == 0 {
                break closest_neighbours
                    .into_iter()
                    .map(|(idx, dist)| (&level.graph[idx], dist))
                    .map(|(graph_node, dist)| (&self.nodes[graph_node.node_idx], dist))
                    .collect();
            }

            let closest_graph_node_idx = closest_neighbours[0].0;
            let closest_graph_node = &level.graph[closest_graph_node_idx];

            if let Some(child) = closest_graph_node.child {
                start_graph_node_idx = child;
            }

            level_idx -= 1;
        }
    }
}

// Function to generate the exponentially decaying probability distribution
fn generate_probabilities<const N: usize>(lambda: f64) -> [f64; N] {
    let mut probabilities: [MaybeUninit<f64>; N] = unsafe { MaybeUninit::uninit().assume_init() };

    // Generate probabilities for each level using the exponential decay
    for (level_idx, level) in probabilities.iter_mut().enumerate() {
        let prob = lambda.exp() * (-lambda * level_idx as f64).exp();
        unsafe {
            level.as_mut_ptr().write(prob);
        }
    }

    // Normalize probabilities so they sum to 1
    let total_sum: f64 = unsafe { probabilities.iter().map(|p| p.assume_init_ref()).sum() };
    for prob in probabilities.iter_mut() {
        unsafe {
            prob.as_mut_ptr().write(prob.assume_init() / total_sum);
        }
    }

    // Convert from MaybeUninit to [f64; N]
    unsafe { std::mem::transmute_copy(&probabilities) }
}
