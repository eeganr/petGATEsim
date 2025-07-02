#!/bin/bash

python3 setup.py build
python3 -m pip install . --user



# g++ -I /share/software/user/open/python/3.12.1/include -I ~/.local/lib/python3.12/site-packages/pybind11/include -o estimations estimations.cpp -lpython3.12

