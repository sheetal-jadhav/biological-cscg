# biological-cscg

**Towards a biologically plausible Clone-Structured Cognitive Graph (CSCG)**

> 🚧 Ongoing project. Code, documentation and usage instructions will be added as the work develops.

## Motivation

The hippocampus turns a stream of ambiguous sensory observations into a cognitive map. The same location can look different depending on where the animal came from or where it is going (splitter cells), and different places can look the same. Clone-Structured Cognitive Graphs (George et al., 2021; Raju et al., 2024) explain many of these phenomena with one idea: each observation is represented by several *clones*, and learning the transitions between clones resolves the ambiguity. Place cells, splitter cells, remapping and reuse of a learned map in a new environment all follow from it.

CSCGs, however, are learned with expectation–maximisation over a hidden Markov model, and it is not clear how a neural circuit could do this. Two lines of work suggest it might be possible:

- Spiking winner-take-all circuits with STDP can approximate online hidden Markov model learning (Kappel et al., 2014).
- A Hebbian recurrent network of this kind reproduces the orthogonalised "state machine" that forms in CA1 as mice learn a task (Sun et al., 2025). Latent representations in a hippocampal network model co-evolve with behavioural exploration (Cone & Clopath, 2024).

This project asks how far such biologically grounded learning rules can be pushed towards a full CSCG. Can a network recruit new clones when an observation turns out to be ambiguous, learn action-dependent transitions, form splitter cells and reuse a learned map in a new environment? The long-term aim is a biophysically detailed hippocampal microcircuit. In it, input-timing-dependent plasticity (ITDP; Dudman et al., 2007) gates recruitment of silent neurons as clones, and behavioural-timescale synaptic plasticity (BTSP; Bittner et al., 2017) stabilises them into context-specific representations.

## References

- George, D., et al. (2021). Clone-structured graph representations enable flexible learning and vicarious evaluation of cognitive maps. *Nature Communications*, 12, 2392. [doi:10.1038/s41467-021-22559-5](https://doi.org/10.1038/s41467-021-22559-5)
- Raju, R. V., et al. (2024). Space is a latent sequence: A theory of the hippocampus. *Science Advances*. [PMC11290523](https://pmc.ncbi.nlm.nih.gov/articles/PMC11290523/)
- Kappel, D., Nessler, B., & Maass, W. (2014). STDP installs in winner-take-all circuits an online approximation to hidden Markov model learning. *PLoS Computational Biology*, 10(3), e1003511. [doi:10.1371/journal.pcbi.1003511](https://doi.org/10.1371/journal.pcbi.1003511)
- Sun, W., et al. (2025). Learning produces an orthogonalized state machine in the hippocampus. *Nature*, 640. [doi:10.1038/s41586-024-08548-w](https://doi.org/10.1038/s41586-024-08548-w)
- Cone, I., & Clopath, C. (2024). Latent representations in hippocampal network model co-evolve with behavioral exploration of task structure. *Nature Communications*, 15, 687. [doi:10.1038/s41467-024-44871-6](https://doi.org/10.1038/s41467-024-44871-6)
- Dudman, J. T., Tsay, D., & Siegelbaum, S. A. (2007). A role for synaptic inputs at distal dendrites: instructive signals for hippocampal long-term plasticity. *Neuron*, 56(5), 866–879. [PMC2179894](https://pmc.ncbi.nlm.nih.gov/articles/PMC2179894/)
- Bittner, K. C., et al. (2017). Behavioral time scale synaptic plasticity underlies CA1 place fields. *Science*, 357(6355), 1033–1036. [doi:10.1126/science.aan3846](https://doi.org/10.1126/science.aan3846)
