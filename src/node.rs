use arrayvec::ArrayVec;
use index_vec::define_index_type;

define_index_type! {
    pub struct NodeIdx = usize;
}

define_index_type! {
    pub struct GraphNodeIdx = usize;
}

pub type Vector<const DIM: usize> = [f32; DIM];
pub type Neighbour = (GraphNodeIdx, f32);

#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct Node<const DIM: usize> {
    pub idx: NodeIdx,
    #[cfg_attr(feature = "serde", serde(with = "serde_big_array::BigArray"))]
    pub vec: Vector<DIM>,
}

#[derive(Debug, Clone, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(
    feature = "rkyv",
    derive(rkyv::Archive, rkyv::Serialize, rkyv::Deserialize)
)]
pub struct GraphNode<const MAX_NEIGHBOURS: usize = 20> {
    pub node_idx: NodeIdx,
    // TODO: use a linked list
    pub neighbours: ArrayVec<Neighbour, MAX_NEIGHBOURS>,
    pub parent: Option<GraphNodeIdx>,
    pub child: Option<GraphNodeIdx>,
}
