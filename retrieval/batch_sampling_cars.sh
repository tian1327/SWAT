#!/bin/bash

python sample_retrieval.py --prefix Random500 --num_samples 500 --sampling_method Random --dataset stanford_cars

python sample_retrieval.py --prefix T2I500 --num_samples 500 --sampling_method T2I-rank --dataset stanford_cars

python sample_retrieval.py --prefix I2I500 --num_samples 500 --sampling_method I2I-rank --dataset stanford_cars

python sample_retrieval.py --prefix I2T500 --num_samples 500 --sampling_method I2T-rank --dataset stanford_cars

python sample_retrieval.py --prefix I2T100 --num_samples 100 --sampling_method I2T-rank --dataset stanford_cars

python sample_retrieval.py --prefix I2T20 --num_samples 20 --sampling_method I2T-rank --dataset stanford_cars

python sample_retrieval.py --prefix T2T10 --num_samples 10 --sampling_method T2T-rank --dataset stanford_cars

