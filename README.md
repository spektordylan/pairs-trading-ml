# Pairs Trading Strategy Research

This project was originally completed for CSCI 4022 as an experimental statistical-arbitrage pipeline combining dimensionality reduction, graph-based pair selection, regime detection, pattern matching, and supervised learning. 

**Note:** The original implementation was developed as a course project and is currently being revisited with a stronger focus on methodological rigor, causal evaluation, and portfolio accounting. The original notebook and report are preserved for reference, while a revised baseline is being developed separately.

The selection of techniques in the original pipeline was influenced in part by the methodological requirements of the course, so some components were included to satisfy project constraints rather than because they were ultimately the strongest choice for the trading problem.

- **Original implementation:** `07-original-pipeline.ipynb`
- **Original project report:** [PDF](https://spektordylan.github.io/doc/CSCI_4022_Project.pdf)
- **Revised pipeline (WIP):** `08-revised-pipeline.ipynb`

## Original Methodology

The original pipeline used historical S&P 500 equity data and proceeded through several stages:

1. **Dimensionality reduction:** PCA was used to separate broad common variation from asset-specific behavior.
2. **Pair screening:** A weighted PageRank procedure was used to reduce the candidate universe before pairwise statistical testing.
3. **Spread modeling:** Candidate pairs were modeled using relative-price behavior and mean-reversion signals.
4. **Regime detection:** Gaussian Mixture Models were used to identify distinct spread regimes.
5. **Pattern matching:** MinHash-based similarity search was used to identify historical spread patterns resembling current conditions.
6. **Signal classification:** A Random Forest classifier combined derived features into trading signals.
7. **Evaluation:** The strategy was evaluated over chronological training, validation, and test periods.

The current revision is focused on auditing and improving the statistical assumptions behind pair construction, eliminating potential sources of look-ahead bias, and rebuilding portfolio returns from explicit long-short positions rather than relying on proxy spread-level P&L.