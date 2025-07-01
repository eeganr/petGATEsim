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

struct Window {
    double time1; // Garbage window default
    double time2;
    
    Window(double t1, double t2) {
        time1 = t1;
        time2 = t2;
    }

    Window() { // Garbage constructor
        time1 = -1;
        time2 = -1;
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
    double window_start = -2 * TAU;
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
        np.array singles_count - [detector index]:[number of singles]
        np.array prompts_count - [detector index]:[number of prompts]
        np.array detectors - NumPy array of detectors present
        double L - result of root finding
        double S - overall singles rate for machine
        double TAU - time coincidence window
        double TIME - total data gathering time
    
    Returns:
        2d numpy array providing the SP rate for each LOR
*/
py::array_t<double> sp_rates(py::array_t<int> singles_count, py::array_t<int> prompts_count, py::array_t<int> detectors, double L, double S, double TAU, double TIME) {
    auto buf = detectors.request();
    int *det = (int*) buf.ptr;

    auto singles_buf = singles_count.request();
    int *singles_ptr = (int*) singles_buf.ptr;

    auto prompts_buf = prompts_count.request();
    int *prompts_ptr = (int*) prompts_buf.ptr;

    int DETS = buf.size; // Number of detectors
    
    py::array_t<double> rates = py::array_t<double>(buf.size * buf.size);
    auto rates_buf = rates.request();
    double *rates_ptr = (double*) rates_buf.ptr;

    double coeff = (2 * TAU * exp(-(L + S)*TAU))/(pow(1 - 2 * L * TAU, 2));

    for (int i = 0; i < DETS; i++) {
        for (int j = i; j < DETS; j++) {
            double P_i = prompts_ptr[det[i]] / TIME;
            double P_j = prompts_ptr[det[j]] / TIME;
            double S_i = singles_ptr[det[i]] / TIME;
            double S_j = singles_ptr[det[j]] / TIME;

            double i_term = S_i - exp((L + S)*TAU) * P_i;
            double j_term = S_j - exp((L + S)*TAU) * P_j;

            rates_ptr[i * DETS + j] = coeff * i_term * j_term;
            rates_ptr[j * DETS + i] = coeff * i_term * j_term;
        }
    }

    rates.resize({DETS, DETS});
    return rates;
}

py::array_t<int> singles_counts(py::array_t<int> detector_occurances, int max_detector_index) {
    auto buf = detector_occurances.request();
    int *ptr = (int*) buf.ptr;
    
    py::object np = py::module_::import("numpy");
    py::array_t<int> result = np.attr("zeros")(max_detector_index + 1);
    auto result_buf = result.request();
    int *result_ptr = (int*) result_buf.ptr;

    for (int i = 0; i < buf.size; i++) {
        result_ptr[ptr[i]]++;
    }

    return result;
}

py::array_t<int> prompts_counts(py::array_t<int> det1_hits, py::array_t<int> det2_hits, int max_detector_index) {
    auto buf1 = det1_hits.request();
    int *ptr1 = (int*) buf1.ptr;
    auto buf2 = det2_hits.request();
    int *ptr2 = (int*) buf2.ptr;
    
    py::object np = py::module_::import("numpy");
    py::array_t<int> result = np.attr("zeros")(max_detector_index + 1);
    auto result_buf = result.request();
    int *result_ptr = (int*) result_buf.ptr;

    for (int i = 0; i < buf1.size; i++) {
        result_ptr[ptr1[i]]++;
        result_ptr[ptr2[i]]++;
    }

    return result;
}

int delayed_window(py::array_t<double> times, double TAU, double DELAY) {
    queue<Window> delays; 
    auto buf = times.request();
    double *ptr = (double*) buf.ptr;

    int dw_estimate = 0;
    int window_start = - 2 * TAU;

    for (int i = 0; i < buf.size; i++) {

        // Part 1: Handling of future delay windows
        // queue future delay window only if we're not already in a coincidence window
        if (ptr[i] - window_start >= TAU) {
            delays.push(Window(ptr[i] + DELAY, ptr[i] + DELAY + TAU));
            window_start = ptr[i]; // reset current coincidence window
        }

        // Part 2: Handling current time relative to past delay window(s)
        Window w;
        while (!delays.empty()) { // past these window(s) already
            w = delays.front();
            if (w.time2 < ptr[i]) {
                delays.pop();
            }   
            else {
                break; // within or before soonest window
            }
        }
        // time1 != time2 indicates non-garbage window
        if ((w.time1 != w.time2) && (w.time1 < ptr[i])) { // within a delay window
            dw_estimate++;
            delays.pop();
        }
    }
    return dw_estimate;
}

PYBIND11_MODULE(randoms, m) {
    m.def("bundle_coincidences", &bundle_coincidences, "Bundles coincidences",
        pybind11::arg("times"),
        pybind11::arg("energies"),
        pybind11::arg("TAU")
    );
    m.def("sp_rates", &sp_rates, "Calculates sp rates",
        pybind11::arg("singles_count"),
        pybind11::arg("prompts_count"),
        pybind11::arg("detectors"),
        pybind11::arg("L"),
        pybind11::arg("S"),
        pybind11::arg("TAU"),
        pybind11::arg("TIME")
    );
    m.def("singles_counts", &singles_counts, "calculates singles counts per detector",
        pybind11::arg("detector_occurances"),
        pybind11::arg("max_detector_index")
    );
    m.def("prompts_counts", &prompts_counts, "calculates prompts counts per detector",
        pybind11::arg("det1_hits"),
        pybind11::arg("det2_hits"),
        pybind11::arg("max_detector_index")
    );
    m.def("delayed_window", &delayed_window, "calculates dw estimate", 
        pybind11::arg("times"),
        pybind11::arg("TAU"),
        pybind11::arg("DELAY")
    );
}