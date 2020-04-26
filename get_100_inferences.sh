#!/bin/bash

for i in $(seq 1 100); do
    echo "{$i}-th inference is on going"
    python LSTM.py
done

echo "All Infrence processes are done"
