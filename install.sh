#!/usr/bin/env bash
set -e

# chemtrain
git clone https://github.com/tummfm/chemtrain.git
cd chemtrain
git checkout 8e05f5a18e2ee09e2f28e5268e47ffa8628ed4b6
pip install -e ".[all,docs,test]"
cd ..

# mace-jax (patched)
git clone https://github.com/ACEsuit/mace-jax.git
cd mace-jax
git checkout 7e9d467d1701290b6606a20ff2c625c27e973254
sed -i 's/find:/find_namespace:/g' setup.cfg
pip install .
cd ..

# remaining deps
pip install -e .