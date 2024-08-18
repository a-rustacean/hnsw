use crate::{
    distance::DistanceFunction,
    node::{GraphNode, GraphNodeIdx, Neighbour, Node, NodeIdx, Vector},
};
use arrayvec::ArrayVec;
use index_vec::IndexVec;
use std::{cmp::Ordering, fmt, marker::PhantomData};

#[derive(Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
pub struct Level<const DIM: usize, DF: DistanceFunction<DIM>, const MAX_NEIGHBOURS: usize = 20> {
    pub graph: IndexVec<GraphNodeIdx, GraphNode<MAX_NEIGHBOURS>>,
    #[cfg_attr(feature = "serde", serde(skip_serializing, default))]
    _marker: PhantomData<DF>,
}

impl<const DIM: usize, DF: DistanceFunction<DIM>, const MAX_NEIGHBOURS: usize> fmt::Debug
    for Level<DIM, DF, MAX_NEIGHBOURS>
{
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Level").field("graph", &self.graph).finish()
    }
}

impl<const DIM: usize, DF: DistanceFunction<DIM>, const MAX_NEIGHBOURS: usize> Default
    for Level<DIM, DF, MAX_NEIGHBOURS>
{
    fn default() -> Self {
        Self {
            graph: IndexVec::new(),
            _marker: PhantomData,
        }
    }
}

impl<const DIM: usize, DF: DistanceFunction<DIM>, const MAX_NEIGHBOURS: usize>
    Level<DIM, DF, MAX_NEIGHBOURS>
{
    pub fn nearest_neighbour_search_starting_from(
        &self,
        query: &Vector<DIM>,
        nodes: &IndexVec<NodeIdx, Node<DIM>>,
        mut current_graph_node_idx: GraphNodeIdx,
    ) -> Vec<Neighbour> {
        let mut current_graph_node = &self.graph[current_graph_node_idx];
        let current_node = &nodes[current_graph_node.node_idx];
        let mut current_dist = DF::distance(&current_node.vec, query);

        'outer: loop {
            let mut neighbours = Vec::with_capacity(current_graph_node.neighbours.len() + 1);
            neighbours.push((current_graph_node_idx, current_dist));
            for neighbour_graph_node_idx in current_graph_node
                .neighbours
                .iter()
                .map(|(id, _)| id)
                .cloned()
            {
                let neighbour_graph_node = &self.graph[neighbour_graph_node_idx];
                let neighbour_node = &nodes[neighbour_graph_node.node_idx];
                let neighbour_dist = DF::distance(&neighbour_node.vec, query);

                if neighbour_dist < current_dist {
                    current_graph_node_idx = neighbour_graph_node_idx;
                    current_graph_node = neighbour_graph_node;
                    current_dist = neighbour_dist;
                    continue 'outer;
                }

                neighbours.push((neighbour_graph_node_idx, neighbour_dist));
            }

            neighbours
                .sort_unstable_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(Ordering::Equal));

            return neighbours;
        }
    }

    pub fn nearest_neighbour_search(
        &self,
        query: &Vector<DIM>,
        nodes: &IndexVec<NodeIdx, Node<DIM>>,
    ) -> Vec<Neighbour> {
        if self.graph.is_empty() {
            return Vec::new();
        }
        self.nearest_neighbour_search_starting_from(query, nodes, GraphNodeIdx::new(0))
    }

    pub fn insert_neighbour(&mut self, graph_node_idx: GraphNodeIdx, neighbour: Neighbour) {
        let graph_node = &mut self.graph[graph_node_idx];
        let mut neighbours = graph_node.neighbours.to_vec();

        let mut i = 1;
        let insert_idx = loop {
            if i >= neighbours.len() {
                break neighbours.len();
            }
            if neighbours[i - 1].1 < neighbour.1 && neighbour.1 < neighbours[i].1 {
                break i;
            }
            i += 1;
        };
        neighbours.insert(insert_idx, neighbour);
        neighbours.truncate(MAX_NEIGHBOURS);
        graph_node.neighbours = ArrayVec::from_iter(neighbours);
    }

    pub fn make_edges_double_sided(
        &mut self,
        graph_node_idx: GraphNodeIdx,
        neighbours: Vec<Neighbour>,
    ) {
        for (neighbour_graph_node_idx, dist) in neighbours {
            self.insert_neighbour(neighbour_graph_node_idx, (graph_node_idx, dist));
        }
    }
}
