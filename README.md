# PulsatileModelLearning

A Julia package for learning and analyzing pulsatile signaling models in T Cell Receptor signaling dynamics.
See associated manuscript Allard, "Improvement in model flexibility reveals a minimal signalling pathway that explains T cell responses to pulsatile stimuli" on [biorxiv](https://www.biorxiv.org/content/10.1101/2025.08.28.672930v1).

This project implements computational models to understand how cells process pulsatile (on-off) signals. The package provides:

- **Multiple model architectures**: Linear models, incoherent feedforward loops (IFFL), and flexible function models
- **Optimization algorithms**: Differential evolution, CMA-ES, and Rowan's simplex methods
- **Comprehensive analysis**: Frequency response analysis, time series visualization, and parameter optimization
- **Experimental data integration**: Analysis against CD69 expression data from [Harris et al. (2020)](https://www.embopress.org/doi/full/10.15252/msb.202010091)


## Quick Start

1. Set up the Julia environment
   ```bash
   julia --project=PulsatileModelLearning -e "using Pkg; Pkg.instantiate()"
   ```
   
   This will install all required dependencies listed in `PulsatileModelLearning/Project.toml`.

2. Run a short end-to-end learning and analysis run

   ```bash
   ./scripts/short_run_end2end.sh
   ```

This script will:
1. **Classical Learning**: Fit a classical model using differential evolution and simplex optimization
2. **Classical Analysis**: Generate frequency response plots and analyze results
3. **Corduroy Learning**: Fit a flexible model using the corduroy algorithm (CMA-ES + simplex alternation)
4. **Corduroy Analysis**: Generate comprehensive analysis plots and time series visualizations

**Expected runtime:** ~5 minutes on an M1 MacBook.

**Output location:** Results are saved in `experiments/YYMMDD/` where `YYMMDD` is today's date:
- `experiments/YYMMDD/data/` - Model parameters and optimization results (.jld2 files)
- `experiments/YYMMDD/plots/` - Analysis figures (.pdf files)

### Project Structure

```
PulsatileModelLearning/
├── PulsatileModelLearning/          # Main Julia package
│   ├── src/                         # Source code
│   │   ├── models/                  # Model definitions
│   │   └── learning_protocols/      # Optimization algorithms
│   └── Project.toml                 # Package dependencies
├── notebooks/                       # Analysis scripts
│   ├── learn_classical.jl          # Classical model training
│   ├── learn_corduroy.jl           # Flexi-model training
│   └── analyze_task5a_deep_winner.jl # Results analysis
├── configs/                         # Configuration files
├── data/                           # Experimental data
├── experiments/                    # Generated results (created on first run)
└── scripts/                       # Automation scripts
```

## Available Models

- **Model5-8**: Classical models with fixed functional forms
- **ModelF6-F8**: Flexible models using learnable basis functions
- **MyModel8**: DI-IFFL (Decay-Inhibition Incoherent Feedforward Loop) architecture

