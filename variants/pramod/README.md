# Variant C: Review-Text-Enhanced NCF

**Owner:** Pramod Yadav

## Approach

Neural Collaborative Filtering (NCF) enhanced with sentence embeddings from
review text. A pretrained sentence-transformer encodes review semantics,
which are fused with user/item representations to capture nuanced preferences.

## Why This Fits HotelRec

- ~71% of reviews have meaningful text averaging 125 words
- Text captures preferences not visible in ratings (e.g., "quiet rooms", "friendly staff")
- Frozen sentence-transformer + trainable projection keeps compute manageable

## Status

Phase 2 -- implementation pending.
