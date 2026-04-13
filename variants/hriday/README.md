# Variant A: LightGCN (Graph-Based Collaborative Filtering)

**Owner:** Hriday Ampavatina

## Approach

LightGCN (He et al., SIGIR 2020) builds a bipartite user-item interaction graph
and learns embeddings via simplified graph convolution -- no feature transformation
or nonlinear activation, just neighborhood aggregation across multiple layers.

## Why This Fits HotelRec

- Graph structure naturally captures multi-hop user-item relationships
- LightGCN handles sparse data well by propagating signals through the graph
- Architecturally distinct from teammates' approaches (NeuMF, text-NCF)
- Lightweight and efficient -- no heavyweight GNN machinery

## Key Design Decisions

- Use the same 20-core preprocessed data and evaluation protocol as baselines
- BPR loss for implicit feedback training (consistent with GMF baseline)
- 3 graph convolution layers with learned layer combination weights

## Status

Phase 2 -- implementation pending.
