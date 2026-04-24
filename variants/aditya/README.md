# Variant B: NeuMF with Attention-Weighted Sub-Ratings

**Owner:** Aditya Hegde

## Approach

Neural Matrix Factorization (NeuMF) enhanced with attention over hotel sub-ratings
(Service, Cleanliness, Location, Value, Rooms, Sleep Quality).

The attention layer learns which sub-rating aspects matter most to each user,
enabling personalized weighting of hotel quality dimensions.

## Why This Fits HotelRec

- HotelRec has rich sub-rating metadata that baselines completely ignore
- Different user types weight aspects differently (business travelers vs families)
- NeuMF provides a strong neural CF backbone; sub-rating attention adds interpretability

## Status

Phase 2 -- Implementation plan finalized. Ready for Claude to generate the architecture and training pipeline.

See [PLAN.md](PLAN.md) for the detailed implementation brief and requirements.
