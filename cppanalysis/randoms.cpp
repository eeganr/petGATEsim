#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <queue>
#include <iostream>

using namespace std;

namespace py = pybind11;

struct Single {
    int index;
    double energy;
    bool operator<(const Single& rhs) const {
        return energy < rhs.energy;
    };

    Single(int index, double energy) {
        this->index = index;
        this->energy = energy;
    }
};

py::array_t<double> bundle_coincidences(py::array_t<double> times, py::array_t<double> energies, double TAU) {
    // Request buffer from input NumPy array
    auto buf = times.request();
    auto buf_eng = energies.request();

    cout << "size: " << buf.size << endl;

    double *ptr = (double*) buf.ptr; // used for indexing times
    double *ptr_eng = (double*) buf_eng.ptr; // used for indexing energy
    
    vector<int> coin_indices;
    double window_start = 0;
    vector<int> possibles;

    for (int i = 0; i < buf.size - 1; i++) {
        if (ptr[i] - window_start >= TAU) {
            // process any previously identified coincidences first
            // If there are at least 2 singles in the window, we have a possible coincidence
            if (possibles.size() >= 2) {
                // Take just the two highest energies
                // takeWinnerOfGoods policy
                priority_queue<Single> goods; // Pops in decreasing energy order
                for (int i : possibles) {
                    goods.push(Single(i, ptr_eng[i]));
                }

                coin_indices.push_back(goods.top().index);
                goods.pop();
                coin_indices.push_back(goods.top().index);
            }
            else if (possibles.size() == 2) {
                coin_indices.push_back(possibles[0]);
                coin_indices.push_back(possibles[1]);
            }
            // Reset the window
            possibles.clear();
            possibles.push_back(i);
            window_start = ptr[i];
        }
        else if (ptr[i] - window_start < TAU) {
            possibles.push_back(i);  // Add to coincidence list
        }
    }

    auto result = py::array_t<int>(coin_indices.size());
    auto result_buf = result.request();
    int *result_ptr = (int*) result_buf.ptr;

    for (int i = 0; i < coin_indices.size(); i++) {
        result_ptr[i] = coin_indices[i];
    }

    return result;
}

PYBIND11_MODULE(randoms, m) {
    m.def("bundle_coincidences", &bundle_coincidences, "Bundles coincidences",
          pybind11::arg("times"),
          pybind11::arg("energies"),
          pybind11::arg("TAU"));
}