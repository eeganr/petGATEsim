#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <queue>
#include <iostream>
#include <cmath>

using namespace std;

namespace py = pybind11;


/* Struct for a single used for multi-coincidence processing.
    Constructor Args:
        index - index for single
        energy - energy for single
*/
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


/* Bundles list of singles into coincidences.
    Args:
        times - NumPy array of arrival times
        energies - NumPy array of energies/single
        TAU - time coincidence window
    Return: 
        numpy array of indices of input times involved in coincidences
    Notes:
        times and energies must be of same length. 
        Requires < int bit limit length singles list since indices are stored.
*/
py::array_t<double> bundle_coincidences(py::array_t<double> times, py::array_t<double> energies, double TAU) {
    // Request buffer from input NumPy array
    auto buf = times.request();
    auto buf_eng = energies.request();

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

/* Calculates singles-prompts rate estimate for each LOR.
    Args:
        dict singles_count - [detector index]:[number of singles]
        dict prompts_count - [detector index]:[number of prompts]
        np.array detectors - NumPy array of detectors present
        double L - result of root finding
        double S - overall singles rate for machine
        double TAU - time coincidence window
        double TIME - total data gathering time
    
    Returns:
        2d numpy array providing the SP rate for each LOR
*/
py::array_t<double> sp_rates(py::dict singles_count, py::dict prompts_count, py::array_t<int> detectors, double L, double S, double TAU, double TIME) {
    auto buf = detectors.request();
    double *det = (double*) buf.ptr;
    int DETS = buf.size; // Number of detectors
    
    py::array_t<double> rates = py::array_t<double>(buf.size * buf.size);
    auto rates_buf = rates.request();
    double *rates_ptr = (double*) rates_buf.ptr;

    double coeff = (2 * TAU * exp(-(L + S)*TAU))/(pow(1 - 2 * L * TAU, 2));

    for (int i = 0; i < DETS; i++) {
        for (int j = i; j < DETS; j++) {
            double P_i = py::cast<double>(prompts_count.attr("get")(det[i], 0));
            double P_j = py::cast<double>(prompts_count.attr("get")(det[j], 0));
            double S_i = py::cast<double>(singles_count.attr("get")(det[i], 0));
            double S_j = py::cast<double>(singles_count.attr("get")(det[j], 0));

            double i_term = S_i - exp((L + S)*TAU) * P_i;
            double j_term = S_j - exp((L + S)*TAU) * P_j;

            rates_ptr[i * DETS + j] = coeff * i_term * j_term;
            rates_ptr[j * DETS + i] = coeff * i_term * j_term;
        }
    }

    rates.resize({DETS, DETS});
    return rates;
}

PYBIND11_MODULE(randoms, m) {
    m.def("bundle_coincidences", &bundle_coincidences, "Bundles coincidences",
          pybind11::arg("times"),
          pybind11::arg("energies"),
          pybind11::arg("TAU"));

    m.def("sp_rates", &sp_rates, "Calculates sp rates",
          pybind11::arg("singles_count"),
          pybind11::arg("prompts_count"),
          pybind11::arg("detectors"),
          pybind11::arg("L"),
          pybind11::arg("S"),
          pybind11::arg("TAU"),
          pybind11::arg("TIME")
          );
}