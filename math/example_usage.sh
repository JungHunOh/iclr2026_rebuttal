#!/bin/bash

# Example usage of the YAML-based training system

# Run Gemma base method
echo "Running Gemma base method..."
python3 run.py configs/gemma_base.yaml --gpu 0,1

# Run Gemma LoRaGA method
echo "Running Gemma LoRaGA method..."
python3 run.py configs/gemma_loraga.yaml --gpu 0,1

# Run Llama3 base method
echo "Running Llama3 base method..."
python3 run.py configs/llama3_base.yaml --gpu 0,1

# Run Llama3 LoRaPro method
echo "Running Llama3 LoRaPro method..."
python3 run.py configs/llama3_lorapro.yaml --gpu 0,1
