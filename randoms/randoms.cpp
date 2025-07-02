#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <queue>
#include <iostream>
#include <cmath>
#include <cassert>

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


/* Struct for a time window used for delayed window estimation.
    Constructor Args:
        t1 - start of the window
        t2 - end of the window
    Notes:
        time1 != time2 indicates a non-garbage window.
*/
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
        np.array<double> times - arrival times of singles
        np.array<double> energies - energies in MeV/single
        TAU - time coincidence window in s
    Return: 
        numpy array of indices of input times involved in coincidences
    Notes:
        times and energies must be of same length. 
        Requires < int bit limit length singles list since indices are stored.
        Currently the multi-coincidence resolution policy is to take the two highest energies
        from the singles in the coincidence window.
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

    for (int i = 0; i < (int) coin_indices.size(); i++) {
        result_ptr[i] = coin_indices[i];
    }

    return result;
}


/* Calculates singles counts per detector.
    Args:
        np.array<int> detector_occurances - at inumber of singles that hit detector of given index
        int max_detector_index - maximum index of the detectors
    Returns:
        numpy array of singles counts per detector
    Notes:
        The output is a 1D NumPy array where the index corresponds to the detector index.
*/
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


/* Calculates prompts counts per detector.
    Args:
        np.array<int> det1_hits - hits for detector 1
        np.array<int> det2_hits - hits for detector 2
        int max_detector_index - maximum index of the detectors
    Returns:
        numpy array of prompts counts per detector
    Notes:
        The output is a 1D NumPy array where the index corresponds to the detector index.
        This function assumes that det1_hits and det2_hits are of the same length since they come from coincidences.
        It counts how many times each detector was involved in a prompt coincidence.
*/
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


/* Calculates singles-prompts rate estimate for each LOR.
    Args:
        np.array<int> singles_count - [detector index]:[number of singles]
        np.array<int> prompts_count - [detector index]:[number of prompts]
        int max_detector_index - the highest numerical detector index
        double L - result of root finding
        double S - overall singles rate for machine
        double TAU - time coincidence window
        double TIME - total data gathering time
    
    Returns:
        2d numpy array providing the SP rate for each LOR
*/
py::array_t<double> sp_rates(py::array_t<int> singles_count, py::array_t<int> prompts_count, int max_detector_index, double L, double S, double TAU, double TIME) {
    auto singles_buf = singles_count.request();
    int *singles_ptr = (int*) singles_buf.ptr;

    auto prompts_buf = prompts_count.request();
    int *prompts_ptr = (int*) prompts_buf.ptr;

    int DETS = max_detector_index + 1; // Number of detectors
    
    py::array_t<double> rates = py::array_t<double>(DETS * DETS);
    auto rates_buf = rates.request();
    double *rates_ptr = (double*) rates_buf.ptr;

    double coeff = (2 * TAU * exp(-(L + S)*TAU))/(pow(1 - 2 * L * TAU, 2));

    for (int i = 0; i < DETS; i++) {
        for (int j = i + 1; j < DETS; j++) {
            double P_i = prompts_ptr[i] / TIME;
            double P_j = prompts_ptr[j] / TIME;
            double S_i = singles_ptr[i] / TIME;
            double S_j = singles_ptr[j] / TIME;

            double i_term = S_i - exp((L + S)*TAU) * P_i;
            double j_term = S_j - exp((L + S)*TAU) * P_j;

            rates_ptr[i * DETS + j] = coeff * i_term * j_term;
            rates_ptr[j * DETS + i] = coeff * i_term * j_term;
        }
    }

    rates.resize({DETS, DETS});
    return rates;
}


/* Calculates delayed window estimate.
    Args:
        np.array<double> times - NumPy array of arrival times
        double TAU - time coincidence window
        double DELAY - delay for the delayed window
    Returns:
        int - number of delayed windows detected
    Notes:
        This function uses a queue to manage the delayed windows and counts 
        how many times the current time falls within a delayed window. Currently
        the multi-coincidence policy is to count each delayed window with a hit only once
        regardless of the number of hits.
*/
int delayed_window(py::array_t<double> times, double TAU, double DELAY) {
    queue<Window> delays; 
    auto buf = times.request();
    double *ptr = (double*) buf.ptr;

    int dw_estimate = 0;
    double window_start = - 2 * TAU;

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


/* Calculates singles-rates estimate for each LOR.
    Args:
        np.array<int> singles_count - [detector index]:[number of singles]
        int max_detector_index - maximum index of the detectors
        double TAU - time coincidence window
        double TIME - total data gathering time
    
    Returns:
        2d numpy array providing the SR rate for each LOR
    Notes:
        The output is a 2D NumPy array where the indices correspond to the detector indices.
*/
py::array_t<double> sr_rates(py::array_t<int> singles_count, int max_detector_index, double TAU, double TIME) {
    auto singles_buf = singles_count.request();
    int *singles_ptr = (int*) singles_buf.ptr;
    int DETS = max_detector_index + 1;

    py::object np = py::module_::import("numpy");
    py::array_t<double> result = np.attr("zeros")(DETS * DETS);
    auto result_buf = result.request();
    double *result_ptr = (double*) result_buf.ptr;

    for (int i = 0; i < DETS; i++) {
        for (int j = i + 1; j < DETS; j++) {
            double estimate = 2 * TAU * singles_ptr[i] / TIME * singles_ptr[j] / TIME;
            result_ptr[i * DETS + j] = estimate;
            result_ptr[j * DETS + i] = estimate;
        }
    }

    result.resize({DETS, DETS});

    return result;
}


/* Calculates actual times from coarse/raw times
    Args:
        np.array<int> coarse - coarse

*/
py::array_t<double> get_times(py::array_t<int> coarse, py::array_t<int> fine, double coarse_t, double fine_t) {
    auto coarse_buf = coarse.request();
    int *coarse_ptr = (int*) coarse_buf.ptr;
    auto fine_buf = fine.request();
    int *fine_ptr = (int*) fine_buf.ptr;

    auto result = py::array_t<double>(coarse_buf.size);
    auto result_buf = result.request();
    double *result_ptr = (double*) result_buf.ptr;

    int last_reset = 0; // coarse times wrap around to 0 after 2^15

    const int COARSE_RESET = pow(2, 15);

    result_ptr[0] = coarse_ptr[0] * coarse_t + fine_ptr[0] * fine_t;

    try {
        for (int i = 1; i < coarse_buf.size; i++) {
            if (coarse_ptr[i] < coarse_ptr[i-1]) {
                last_reset++;
            }
            result_ptr[i] = 
                (coarse_ptr[i] * coarse_t)
                + (COARSE_RESET * coarse_t * last_reset)
                + fine_ptr[i] * fine_t;
                
            if (result_ptr[i] < 0) {
                cout << last_reset << " and " << COARSE_RESET << " and " << coarse_ptr[i] << " and " << fine_ptr[i] << endl;
                cout << result_ptr[i] << endl;
                cout << coarse_t << " and " << fine_t << endl;
                throw overflow_error(" integer overflow! ");
                break;
            }
        }
    }
    catch(const overflow_error& e) {
        cout << "integer overflow, file likely too long" << endl;
    }

    return result;
}


PYBIND11_MODULE(randoms, m) {
    m.def("bundle_coincidences", &bundle_coincidences, "Bundles coincidences",
        py::arg("times"),
        py::arg("energies"),
        py::arg("TAU")
    );
    m.def("singles_counts", &singles_counts, "calculates singles counts per detector",
        py::arg("detector_occurances"),
        py::arg("max_detector_index")
    );
    m.def("prompts_counts", &prompts_counts, "calculates prompts counts per detector",
        py::arg("det1_hits"),
        py::arg("det2_hits"),
        py::arg("max_detector_index")
    );
    m.def("sp_rates", &sp_rates, "Calculates sp rates",
        py::arg("singles_count"),
        py::arg("prompts_count"),
        py::arg("max_detector_index"),
        py::arg("L"),
        py::arg("S"),
        py::arg("TAU"),
        py::arg("TIME")
    );
    m.def("delayed_window", &delayed_window, "calculates dw estimate", 
        py::arg("times"),
        py::arg("TAU"),
        py::arg("DELAY")
    );
    m.def("sr_rates", &sr_rates, "calculates sr estimate",
        py::arg("singles_count"),
        py::arg("max_detector_index"),
        py::arg("TAU"),
        py::arg("TIME")
    );
    m.def("get_times", &get_times, "gives raw time from coarse and fine",
        py::arg("coarse"),
        py::arg("fine"),
        py::arg("coarse_t"),
        py::arg("fine_t")
    );
}