#!/bin/bash
python3 meanweight.py ../../results/0.ee2.syn > x.syn
echo "plot 'x.syn' w l" | gnuplot -p
