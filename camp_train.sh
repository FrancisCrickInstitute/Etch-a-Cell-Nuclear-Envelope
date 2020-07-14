#!/bin/sh
ml purge
ml Singularity
singularity exec --nv organelle-pipeline_latest.sif python3 /em/run_training_pipeline.py --ignore_steps 1 --restart false