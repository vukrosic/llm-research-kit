"""
Frontier Architecture Research System
=======================================
Beyond-transformer architecture search for the next paradigm shift
in sequence modeling for language.

Usage:
    # List all registered architectures
    python -m frontier.experiments.run_frontier --list

    # Generate next batch of experiments
    python -m frontier.evolution.architect --batch-size 15

    # Run all pending experiments
    python -m frontier.experiments.run_frontier

    # Run a specific experiment
    python -m frontier.experiments.run_frontier --exp ssm_mamba_baseline

    # Compare results
    python -m frontier.analysis.compare
"""
