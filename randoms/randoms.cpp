#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <vector>
#include <queue>
#include <iostream>
#include <cmath>
#include <string>
#include <iostream>
#include <fstream>
#include <cstdint>
#include <filesystem>
#include <random>
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

using namespace std;

namespace py = pybind11;

constexpr int NUM_VOL_IDS = 6;
constexpr long SPD_OF_LIGHT = 299792458000; // mm/s
constexpr int BUFFER_SIZE = 10; // number of records to keep in priority queue for chronology
constexpr int MODULES = 16;
constexpr int LORS = 120;

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
    vector<int> events;
    
    Window(double t1, double t2, int trigger_index) {
        time1 = t1;
        time2 = t2;
        events = {trigger_index};
    }

    Window() { // Garbage constructor
        time1 = -1;
        time2 = -1;
    }
};


/* Struct for a Record output in the "Singles" category by GATE
*/
struct Record {
    int32_t run;
    int32_t event;
    int32_t srcID;

    double srcX, srcY, srcZ;

    int32_t volIDs[NUM_VOL_IDS];

    double time;
    double Edep;
    double detX, detY, detZ;

    int32_t nComPh, nComDet;
    int32_t nRayPh, nRayDet;

    char phantomCom[8];
    char phantomRay[8];

    /* Returns ID of detector from this record
        Returns:
            int - detector ID
    */
    int id() {
        return volIDs[1] * 6 * 128 + volIDs[3] * 128 + volIDs[4];
    }
};


/* Struct for a time window used for delayed window estimation.
    Constructor Args:
        double t1 - start of the window
        double t2 - end of the window
        Record r - prompt event
    Notes:
        time1 != time2 indicates a non-garbage window.
*/
struct WindowLM {
    double time1; // Garbage window default
    double time2;
    vector<Record> events;
    
    WindowLM(double t1, double t2, Record r) {
        time1 = t1;
        time2 = t2;
        events = {r};
    }

    WindowLM() { // Garbage constructor
        time1 = -1;
        time2 = -1;
    }
};


#pragma pack(push, 1)
struct ListmodeRecord {
    float x1, y1, z1;

    float TOF;

    float unused;

    float x2, y2, z2;

    float crystalID1, crystalID2;
};
#pragma pack(pop)


/* Bundles list of singles into coincidences.
    Args:
        np.array<double> times - arrival times of singles
        TAU - time coincidence window in s
    Return: 
        Tuple 
        (
            int numpy array of indices of input times involved in coincidences,
            int numpy array of multiple coincidences (number of hits involved in each
        )
    Notes:
        times and energies must be of same length. 
        Requires < int bit limit length singles list since indices are stored.
        Currently the multi-coincidence resolution policy is to take the two highest energies
        from the singles in the coincidence window.
*/
py::tuple bundle_coincidences(py::array_t<double> times, double TAU) {
    // Request buffer from input NumPy array
    auto buf = times.request();
    // auto buf_eng = energies.request();

    double *ptr = (double*) buf.ptr; // used for indexing times
    // double *ptr_eng = (double*) buf_eng.ptr; // used for indexing energy
    
    vector<int> coin_indices;
    double window_start = -2 * TAU;
    vector<int> possibles;
    vector<int> multi_coincidences;

    for (int i = 0; i < buf.size - 1; i++) {
        if (ptr[i] - window_start >= TAU) {
            // process any previously identified coincidences first
            // If there are at least 2 singles in the window, we have a possible coincidence
            if (possibles.size() > 2) {

                multi_coincidences.push_back(possibles.size());

                /* takeWinnerOfGoods policy, takes just the two highest energies
                priority_queue<Single> goods; // Pops in decreasing energy order
                for (int i : possibles) {
                    goods.push(Single(i, ptr_eng[i]));
                }

                // Ensure they're stored in the correct chronological order
                int i1 = goods.top().index;
                goods.pop();
                int i2 = goods.top().index;

                coin_indices.push_back(min(i1, i2));
                coin_indices.push_back(max(i1, i2));
                */
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

    auto multis = py::array_t<int>(multi_coincidences.size());
    auto multi_buf = multis.request();
    int *multi_ptr = (int*) multi_buf.ptr;

    for (int i = 0; i < (int) coin_indices.size(); i++) {
        result_ptr[i] = coin_indices[i];
    }
    for (int i = 0; i < (int) multi_coincidences.size(); i++) {
        multi_ptr[i] = multi_coincidences[i];
    }

    return py::make_tuple(result, multis);
}


/* Calculates singles counts per detector.
    Args:
        np.array<int> detector_occurances - number of singles that hit detector of given index
        int num_detectors - maximum index of the detectors
    Returns:
        numpy array of singles counts per detector
    Notes:
        The output is a 1D NumPy array where the index corresponds to the detector index.
*/
py::array_t<int> singles_counts(py::array_t<int> detector_occurances, int num_detectors) {
    auto buf = detector_occurances.request();
    int *ptr = (int*) buf.ptr;
    
    py::object np = py::module_::import("numpy");
    py::array_t<int> result = np.attr("zeros")(num_detectors);
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
        int num_detectors - maximum index of the detectors
    Returns:
        numpy array of prompts counts per detector
    Notes:
        The output is a 1D NumPy array where the index corresponds to the detector index.
        This function assumes that det1_hits and det2_hits are of the same length since they come from coincidences.
        It counts how many times each detector was involved in a prompt coincidence.
*/
py::array_t<int> prompts_counts(py::array_t<int> det1_hits, py::array_t<int> det2_hits, int num_detectors) {
    auto buf1 = det1_hits.request();
    int *ptr1 = (int*) buf1.ptr;
    auto buf2 = det2_hits.request();
    int *ptr2 = (int*) buf2.ptr;
    
    py::object np = py::module_::import("numpy");
    py::array_t<int> result = np.attr("zeros")(num_detectors);
    auto result_buf = result.request();
    int *result_ptr = (int*) result_buf.ptr;

    for (int i = 0; i < buf1.size; i++) {
        result_ptr[ptr1[i]]++;
        result_ptr[ptr2[i]]++;
    }

    return result;
}


/* Calculates coincidences for each LOR.
    Args:
        np.array det1_hits - first detector involved in coincidences
        np.array det2_hits - second detector involved in coincidences, same length as det1_hits
        int num_detectors - maximum index of the detectors
    Returns:
        2d numpy array for coincidences for each LOR, indexed by crystals at end of LOR
*/
py::array_t<int> coincidences_per_lor(py::array_t<int> det1_hits, py::array_t<int> det2_hits, int num_detectors) {
    auto buf1 = det1_hits.request();
    int *ptr1 = (int*) buf1.ptr;
    auto buf2 = det2_hits.request();
    int *ptr2 = (int*) buf2.ptr;

    int DETS = num_detectors;

    py::object np = py::module_::import("numpy");
    py::array_t<int> result = np.attr("zeros")(DETS * DETS);
    auto result_buf = result.request();
    int *result_ptr = (int*) result_buf.ptr;

    for (int i = 0; i < buf1.size; i++) {
        result_ptr[ptr1[i] * DETS + ptr2[i]]++;
        result_ptr[ptr2[i] * DETS + ptr1[i]]++;
    }

    result.resize({DETS, DETS});

    return result;
}


/* Calculates singles-prompts rate estimate for each LOR.
    Args:
        np.array<int> singles_count - [detector index]:[number of singles]
        np.array<int> prompts_count - [detector index]:[number of prompts]
        int num_detectors - the highest numerical detector index
        double L - result of root finding
        double S - overall singles rate for machine
        double TAU - time coincidence window
        double TIME - total data gathering time
    
    Returns:
        2d numpy array providing the SP rate for each LOR
*/
py::array_t<double> sp_rates(py::array_t<int> singles_count, py::array_t<int> prompts_count, int num_detectors, double L, double S, double TAU, double TIME) {
    auto singles_buf = singles_count.request();
    int *singles_ptr = (int*) singles_buf.ptr;

    auto prompts_buf = prompts_count.request();
    int *prompts_ptr = (int*) prompts_buf.ptr;

    int DETS = num_detectors; // Number of detectors
    
    py::object np = py::module_::import("numpy");
    py::array_t<double> rates = np.attr("zeros")(DETS * DETS);
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


py::array_t<double> sp_correction(py::array_t<int> singles_count, int num_detectors, double exp_prod, double TAU, double TIME) {
    auto singles_buf = singles_count.request();
    int *singles_ptr = (int*) singles_buf.ptr;

    int DETS = num_detectors; // Number of detectors
    
    py::object np = py::module_::import("numpy");
    py::array_t<double> result = np.attr("full")(DETS * DETS, exp_prod);
    auto result_buf = result.request();
    double *result_ptr = (double*) result_buf.ptr;

    for (int i = 0; i < DETS; i++) {
        for (int j = i + 1; j < DETS; j++) {
            double S_i = singles_ptr[i] / TIME;
            double S_j = singles_ptr[j] / TIME;

            double factor = exp(-S_i * TAU * TAU / TIME) * exp(-S_j * TAU * TAU / TIME);

            result_ptr[i * DETS + j] /= factor;
            result_ptr[j * DETS + i] /= factor;
        }
    }

    result.resize({DETS, DETS});
    return result;
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
py::array_t<int> dw_rates(py::array_t<double> times, py::array_t<int> detectors, int num_detectors, double TAU, double DELAY) {
    queue<Window> delays; 
    auto buf = times.request();
    double *ptr = (double*) buf.ptr;
    
    auto det_buf = detectors.request();
    int *det_ptr = (int*) det_buf.ptr;

    double window_start = - 2 * TAU; // Set below intentionally

    // Setup results array
    int DETS = num_detectors; // Number of detectors
    py::object np = py::module_::import("numpy");
    py::array_t<int> rates = np.attr("zeros")(DETS * DETS);
    auto rates_buf = rates.request();
    int *rates_ptr = (int*) rates_buf.ptr;

    for (int i = 0; i < buf.size; i++) {

        // Part 1: Handling of future delay windows
        // queue future delay window only if we're not already in a coincidence window
        if (ptr[i] - window_start >= TAU) {
            delays.push(Window(ptr[i] + DELAY, ptr[i] + DELAY + TAU, i));
            window_start = ptr[i]; // reset current coincidence window
        }

        // Part 2: Handling current time relative to past delay window(s)
        Window w;
        while (!delays.empty()) { 
            w = delays.front();
            if (w.time2 < ptr[i]) { // past these window(s) already
                // Process the windows
                if (w.events.size() == 2) {
                    int i = det_ptr[w.events[0]];
                    int j = det_ptr[w.events[1]];
                    rates_ptr[i * DETS + j] ++;
                    rates_ptr[j * DETS + i] ++;
                }
                delays.pop();
            }   
            else {
                break; // within or before soonest window
            }
        }
        // empty events indicates garbage window
        if ((!w.events.empty()) && (w.time1 < ptr[i])) { // within a delay window
            delays.front().events.push_back(i);
        }
    }
    rates.resize({DETS, DETS});
    return rates;
}


/* Calculates singles-rates estimate for each LOR.
    Args:
        np.array<int> singles_count - [detector index]:[number of singles]
        int num_detectors - maximum index of the detectors
        double TAU - time coincidence window
        double TIME - total data gathering time
    
    Returns:
        2d numpy array providing the SR rate for each LOR
    Notes:
        The output is a 2D NumPy array where the indices correspond to the detector indices.
*/
py::array_t<double> sr_rates(py::array_t<int> singles_count, int num_detectors, double TAU, double TIME) {
    auto singles_buf = singles_count.request();
    int *singles_ptr = (int*) singles_buf.ptr;
    int DETS = num_detectors;

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


/* Reads records defined in Record struct from bin file
    Args:
        ifstream in - instream
        Record& rec - record to modify as return value
    Returns:
        bool - whether read was successful
*/
bool read_record(ifstream& in, Record& rec) {
    if (!in.read(reinterpret_cast<char*>(&rec.run), sizeof(int32_t))) return false;
    if (!in.read(reinterpret_cast<char*>(&rec.event), sizeof(int32_t))) return false;
    if (!in.read(reinterpret_cast<char*>(&rec.srcID), sizeof(int32_t))) return false;

    if (!in.read(reinterpret_cast<char*>(&rec.srcX), sizeof(double))) return false;
    if (!in.read(reinterpret_cast<char*>(&rec.srcY), sizeof(double))) return false;
    if (!in.read(reinterpret_cast<char*>(&rec.srcZ), sizeof(double))) return false;

    for (int i = 0; i < NUM_VOL_IDS; ++i) {
        if (!in.read(reinterpret_cast<char*>(&rec.volIDs[i]), sizeof(int32_t))) return false;
    }

    if (!in.read(reinterpret_cast<char*>(&rec.time), sizeof(double))) return false;
    if (!in.read(reinterpret_cast<char*>(&rec.Edep), sizeof(double))) return false;

    if (!in.read(reinterpret_cast<char*>(&rec.detX), sizeof(double))) return false;
    if (!in.read(reinterpret_cast<char*>(&rec.detY), sizeof(double))) return false;
    if (!in.read(reinterpret_cast<char*>(&rec.detZ), sizeof(double))) return false;

    if (!in.read(reinterpret_cast<char*>(&rec.nComPh), sizeof(int32_t))) return false;
    if (!in.read(reinterpret_cast<char*>(&rec.nComDet), sizeof(int32_t))) return false;
    if (!in.read(reinterpret_cast<char*>(&rec.nRayPh), sizeof(int32_t))) return false;
    if (!in.read(reinterpret_cast<char*>(&rec.nRayDet), sizeof(int32_t))) return false;

    if (!in.read(reinterpret_cast<char*>(&rec.phantomCom), 8)) return false;
    if (!in.read(reinterpret_cast<char*>(&rec.phantomRay), 8)) return false;

    return true;
}


/* Processes binary file, extracting all as-read info
    Args:
        string path - path of bin file to be read
        double TAU - coincidence time window
        double DELAY - delay used for delay window
        int num_detectors - max det ID + 1
    Returns:
        np::array<int> singles_count - singles per detector
        np::array<int> prompts_count - prompts per detector
        np::array<int> coin_lor - 2d of coincidences on a given LOR
        np::array<int> dw - 2d of delayed window estimate on LOR
        np::array<int> actuals - 2d of actual # of randoms on LOR
*/
py::tuple read_file(string path, double TAU, double DELAY, int num_detectors) {
    // Open Infile
    ifstream file(path, ios::binary);
    if (!file) {
        cerr << "Error: Could not open file.\n";
        return py::make_tuple(0, 0);
    }

    // Singles Count Array
    py::object np = py::module_::import("numpy");
    py::array_t<int> scount = np.attr("zeros")(num_detectors);
    auto scount_buf = scount.request();
    int *scount_ptr = (int*) scount_buf.ptr;

    // Prompts Count Array
    py::array_t<int> pcount = np.attr("zeros")(num_detectors);
    auto pcount_buf = pcount.request();
    int *pcount_ptr = (int*) pcount_buf.ptr;

    // Local Variables for Coincidence Bundling
    double window_start = -2 * TAU;
    vector<Record> possibles;

    // Variables for Delayed Window
    queue<Window> delays;
    py::array_t<int> dw = np.attr("zeros")(num_detectors * num_detectors);
    auto dw_buf = dw.request();
    int *dw_ptr = (int*) dw_buf.ptr;

    // Actual Randoms
    py::array_t<int> actuals = np.attr("zeros")(num_detectors * num_detectors);
    auto actual_buf = actuals.request();
    int *actual_ptr = (int*) actual_buf.ptr;

    // Coincidences Per LOR
    py::array_t<int> coin_lor = np.attr("zeros")(num_detectors * num_detectors);
    auto coin_lor_buf = coin_lor.request();
    int *coin_lor_ptr = (int*) coin_lor_buf.ptr;

    // Variables for Read Loop
    Record rec;
    int record_count = 0;
    auto chrono = [] (Record a, Record b) {return a.time > b.time;};
    // Buffer to ensure that records read chronologically
    priority_queue<Record, vector<Record>, decltype(chrono)> buffer(chrono);

    for (int i = 0; i < BUFFER_SIZE; i++) { // Reads in BUFFER_SIZE records to queue to start with
        if (!read_record(file, rec)) { // breaks if end of file too soon
            break;
        }
        buffer.push(rec);
    }
    

    // START MAIN LOOP
    while (!buffer.empty()) {
        // Logging read info, updating rec in while statement.
        record_count++;
        if (record_count % 10000000 == 0) {
            cout << "Processed " 
                << record_count / 1000000 
                << " million records." << endl;
            cout << "Sim Time: " << rec.time << endl;
        }

        // Retrieve from Buffer
        rec = buffer.top();
        buffer.pop();

        int detID = rec.id();

        // Count singles
        scount_ptr[detID]++;

        // Coincidence Processing
        if (rec.time - window_start >= TAU) {
            if (possibles.size() > 2) {
                // Do nothing, since multiple coincidence
            }
            else if (possibles.size() == 2) {
                // Add to prompts count array
                int i = possibles[0].id();
                int j = possibles[1].id();
                pcount_ptr[i]++;
                pcount_ptr[j]++;
                // Add to "coincidences per LOR" array
                coin_lor_ptr[i * num_detectors + j]++;
                coin_lor_ptr[j * num_detectors + i]++;
                if (
                    possibles[0].srcX != possibles[1].srcX
                    || possibles[0].srcY != possibles[1].srcY
                    || possibles[0].srcZ != possibles[1].srcZ
                ) {
                    actual_ptr[i * num_detectors + j]++;
                    actual_ptr[j * num_detectors + i]++;
                }

                // TODO: Save to listmode!
            }
            // Create new delayed window prompt
            delays.push(Window(rec.time + DELAY, rec.time + DELAY + TAU, detID));
            // Handle resetting
            possibles.clear();
            possibles.push_back(rec);
            window_start = rec.time;
        }
        else {
            possibles.push_back(rec); 
        }

        // Delayed Window Processing
        Window w;
        while (!delays.empty()) { 
            w = delays.front();
            if (w.time2 < rec.time) { // past these window(s) already
                // Process the windows
                if (w.events.size() == 2) {
                    int i = w.events[0];
                    int j = w.events[1];
                    dw_ptr[i * num_detectors + j]++;
                    dw_ptr[j * num_detectors + i]++;
                }
                delays.pop();
            }   
            else {
                break; // within or before soonest window
            }
        }
        // empty events indicates garbage window
        if ((!w.events.empty()) && (w.time1 < rec.time)) { // within a delay window
            delays.front().events.push_back(detID);
        }

        // Push next event to buffer
        if (read_record(file, rec)) {
            buffer.push(rec);
        }

    } // End Main Loop


    if (file.eof()) {
        cout << "Reached end of file after reading " << record_count << " records.\n";
    } else if (file.fail()) {
        cerr << "File read error occurred!\n";
    }

    coin_lor.resize({num_detectors, num_detectors});
    dw.resize({num_detectors, num_detectors});
    actuals.resize({num_detectors, num_detectors});

    return py::make_tuple(scount, pcount, coin_lor, dw, actuals);
}


/* Processes binary file, extracting all as-read info, writes listmode files
    Args:
        string path - path of bin file to be read
        string outfolder - folder to save output files
        string name - base name for output files
        double TAU - coincidence time window
        double DELAY - delay used for delay window
        int num_detectors - max det ID + 1
    Returns:
        np::array<int> singles_count - singles per detector
        np::array<int> prompts_count - prompts per detector
        np::array<int> coin_lor - 2d of coincidences on a given LOR
        np::array<int> dw - 2d of delayed window estimate on LOR
        np::array<int> actuals - 2d of actual # of randoms on LOR
    Writes:
        listmode file - all coincidences
        listmode file - all delayed coincidences
        listmode file - all actual randoms
*/
py::tuple read_file_lm(string path, string outfolder, string name, double TAU, double DELAY, int num_detectors) {
    ifstream file(path, ios::binary);
    if (!file) {
        cerr << "Error: Could not open infile.\n";
        return py::make_tuple(0, 0);
    }

    ofstream outfile(outfolder + name + ".lm", ios::binary | ios::app);
    if (!outfile) {
        cerr << "Error: Could not open outfile for writing.\n";
        return py::make_tuple(0, 0);
    }

    ofstream delayfile(outfolder + name + "_delay.lm", ios::binary | ios::app);
    if (!delayfile) {
        cerr << "Error: Could not open delayfile for writing.\n";
        return py::make_tuple(0, 0);
    }

    ofstream actfile(outfolder + name + "_actual.lm", ios::binary | ios::app);
    if (!actfile) {
        cerr << "Error: Could not open actfile for writing.\n";
        return py::make_tuple(0, 0);
    }

    // Singles Count Array
    py::object np = py::module_::import("numpy");
    py::array_t<int> scount = np.attr("zeros")(num_detectors);
    auto scount_buf = scount.request();
    int *scount_ptr = (int*) scount_buf.ptr;

    // Prompts Count Array
    py::array_t<int> pcount = np.attr("zeros")(num_detectors);
    auto pcount_buf = pcount.request();
    int *pcount_ptr = (int*) pcount_buf.ptr;

    // Local Variables for Coincidence Bundling
    double window_start = -2 * TAU;
    vector<Record> possibles;

    // Variables for Delayed Window
    queue<WindowLM> delays;
    py::array_t<int> dw = np.attr("zeros")(num_detectors * num_detectors);
    auto dw_buf = dw.request();
    int *dw_ptr = (int*) dw_buf.ptr;

    // Actual Randoms
    py::array_t<int> actuals = np.attr("zeros")(num_detectors * num_detectors);
    auto actual_buf = actuals.request();
    int *actual_ptr = (int*) actual_buf.ptr;

    // Coincidences Per LOR
    py::array_t<int> coin_lor = np.attr("zeros")(num_detectors * num_detectors);
    auto coin_lor_buf = coin_lor.request();
    int *coin_lor_ptr = (int*) coin_lor_buf.ptr;

    // Variables for Read Loop
    Record rec;
    int record_count = 0;
    auto chrono = [] (Record a, Record b) {return a.time > b.time;};
    // Buffer to ensure that records read chronologically
    priority_queue<Record, vector<Record>, decltype(chrono)> buffer(chrono);

    for (int i = 0; i < BUFFER_SIZE; i++) { // Reads in BUFFER_SIZE records to queue to start with
        if (!read_record(file, rec)) { // breaks if end of file too soon
            break;
        }
        buffer.push(rec);
    }

    // START MAIN LOOP
    while (!buffer.empty()) {
        // Logging read info, updating rec in while statement.
        record_count++;
        if (record_count % 10000000 == 0) {
            cout << "Processed " 
                << record_count / 1000000 
                << " million records." << endl;
            cout << "Sim Time: " << rec.time << endl;
        }

        // Retrieve from Buffer
        rec = buffer.top();
        buffer.pop();

        int detID = rec.id();

        // Count singles
        scount_ptr[detID]++;

        // Coincidence Processing
        if (rec.time - window_start >= TAU) {
            if (possibles.size() > 2) {
                // Do nothing, since multiple coincidence
            }
            else if (possibles.size() == 2) {
                // Add to prompts count array
                int i = possibles[0].id();
                int j = possibles[1].id();
                pcount_ptr[i]++;
                pcount_ptr[j]++;
                // Add to "coincidences per LOR" array
                coin_lor_ptr[i * num_detectors + j]++;
                coin_lor_ptr[j * num_detectors + i]++;
                // Write to LM file

                float tof_mm = SPD_OF_LIGHT * (possibles[1].time - possibles[0].time);

                ListmodeRecord outrec;
                if (i < j) {
                    outrec = {
                        (float) possibles[0].detX, (float) possibles[0].detY, (float) possibles[0].detZ, 
                        tof_mm, 0.0, 
                        (float) possibles[1].detX, (float) possibles[1].detY, (float) possibles[1].detZ, 
                        (float) i, (float) j
                    };
                } else {
                    outrec = {
                        (float) possibles[1].detX, (float) possibles[1].detY, (float) possibles[1].detZ, 
                        -tof_mm, 0.0, 
                        (float) possibles[0].detX, (float) possibles[0].detY, (float) possibles[0].detZ, 
                        (float) j, (float) i
                    };
                }

                outfile.write(reinterpret_cast<char*>(&outrec), sizeof(ListmodeRecord));

                // Determine whether it's actually a random
                if (
                    possibles[0].srcX != possibles[1].srcX
                    || possibles[0].srcY != possibles[1].srcY
                    || possibles[0].srcZ != possibles[1].srcZ
                ) {
                    actual_ptr[i * num_detectors + j]++;
                    actual_ptr[j * num_detectors + i]++;

                    actfile.write(reinterpret_cast<char*>(&outrec), sizeof(ListmodeRecord));
                }

            }
            // Create new delayed window prompt
            delays.push(WindowLM(rec.time + DELAY, rec.time + DELAY + TAU, rec));
            // Handle resetting
            possibles.clear();
            possibles.push_back(rec);
            window_start = rec.time;
        }
        else {
            possibles.push_back(rec); 
        }

        // Delayed Window Processing
        WindowLM w;
        while (!delays.empty()) { 
            w = delays.front();
            if (w.time2 < rec.time) { // past these window(s) already
                // Process the windows
                if (w.events.size() == 2) {
                    int i = w.events[0].id();
                    int j = w.events[1].id();
                    dw_ptr[i * num_detectors + j]++;
                    dw_ptr[j * num_detectors + i]++;
                    // Write delay coincidence
                    ListmodeRecord outdelay;
                    float tof_mm = SPD_OF_LIGHT * (w.events[1].time - w.events[0].time - DELAY);
                    if (i < j) {
                        outdelay = {
                            (float) w.events[0].detX, (float) w.events[0].detY, (float) w.events[0].detZ, 
                            tof_mm, 0.0, 
                            (float) w.events[1].detX, (float) w.events[1].detY, (float) w.events[1].detZ, 
                            (float) i, (float) j
                        };
                    } else {
                        outdelay = {
                            (float) w.events[1].detX, (float) w.events[1].detY, (float) w.events[1].detZ, 
                            -tof_mm, 0.0, 
                            (float) w.events[0].detX, (float) w.events[0].detY, (float) w.events[0].detZ, 
                            (float) j, (float) i
                        };
                    }

                    delayfile.write(reinterpret_cast<char*>(&outdelay), sizeof(ListmodeRecord));
                }
                delays.pop();
            }   
            else {
                break; // within or before soonest window
            }
        }
        // empty events indicates garbage window
        if ((!w.events.empty()) && (w.time1 < rec.time)) { // within a delay window
            delays.front().events.push_back(rec);
        }

        // Push next event to buffer
        if (read_record(file, rec)) {
            buffer.push(rec);
        }

    } // End Main Loop


    if (file.eof()) {
        cout << "Reached end of file after reading " << record_count << " records.\n";
    } else if (file.fail()) {
        cerr << "File read error occurred!\n";
    }

    coin_lor.resize({num_detectors, num_detectors});
    dw.resize({num_detectors, num_detectors});
    actuals.resize({num_detectors, num_detectors});

    outfile.close();

    return py::make_tuple(scount, pcount, coin_lor, dw, actuals);
}


/* Histogram of Time-of-Flight (TOF) values 
    Args:
        path: Path to the input file
        abs_max: Absolute maximum TOF value
        num_bins: Number of bins for the histogram
    Returns:
        Histogram as a numpy array
*/
py::array_t<int> hist_tof(string path, int abs_max, int num_bins) {
    ifstream infile(path, ios::binary);
    if (!infile) {
        cerr << "Error: Could not open file.\n";
        return py::make_tuple(0, 0);
    }
    
    int bin_size = abs_max * 2 / num_bins;

    py::object np = py::module_::import("numpy");
    py::array_t<int> out = np.attr("zeros")(num_bins);
    auto out_buf = out.request();
    int *out_ptr = (int*) out_buf.ptr;

    ListmodeRecord rec;
    int read = 0;
    while (infile.read(reinterpret_cast<char*>(&rec), sizeof(ListmodeRecord))) {
        read++;
        if (read % 100000000 == 0) {
            cout << "Read " << read << " records." << endl;
        }
        out_ptr[(int) (rec.TOF / bin_size + num_bins / 2)]++;
    }

    return out;
}


/* Splits large listmode file into smaller chunks by LOR
    Args:
        inpath: Path to the input file
        outpath: Path to the output folder
        name: Base name for output files
        max_detectors: Maximum number of detectors
    Writes:
        listmode file(s) - 1 for each LOR
*/
void split_lm(string inpath, string outpath, string name, int max_detectors) {
    int crystal_per_lor = max_detectors / MODULES;

    ifstream infile(inpath, ios::binary);
    if (!infile) {
        cerr << "Error: Could not open file.\n";
        return;
    }
    
    ofstream outs[MODULES][MODULES];

    int count_same = 0;

    cout << "Making outstreams..." << endl;
    for (int i = 0; i < MODULES; i++) {
        for (int j = i + 1; j < MODULES; j++) {
            outs[i][j] = ofstream(outpath + to_string(i) + '_' + to_string(j) + '_' + name + ".lm", ios::binary);
            if (!outs[i][j]) {
                cerr << "Error: Could not open outfile for writing.\n";
                return;
            }
        }
    }
    cout << "Made outstreams!" << endl;

    int read = 0;

    ListmodeRecord rec;

    while (infile.read(reinterpret_cast<char*>(&rec), sizeof(ListmodeRecord))) {
        read++;
        if (read % 100000000 == 0) {
            cout << "Read " << read << " records." << endl;
        }

        if (rec.crystalID1 > rec.crystalID2) {
            ListmodeRecord copy = rec;
            rec.x1 = rec.x2; rec.y1 = rec.y2; rec.z1 = rec.z2;
            rec.x2 = copy.x1; rec.y2 = copy.y1; rec.z2 = copy.z1;
            rec.TOF *= -1;
            rec.crystalID1 = rec.crystalID2;
            rec.crystalID2 = copy.crystalID1;
        }

        int a = rec.crystalID1 / crystal_per_lor;
        int b = rec.crystalID2 / crystal_per_lor;
        int i = min(a, b);
        int j = max(a, b);
        if (i == j) {
            count_same++;
        } else {
            outs[i][j].write(reinterpret_cast<char*>(&rec), sizeof(ListmodeRecord));
        }
    }
    cout << "Counted this many on same det: " << count_same << endl;
}


// void combine_lm(string inpath, string outpath) {
//     ifstream infile(inpath, ios::binary);
//     if (!infile) {
//         cerr << "Error: Could not open infile for reading.\n";
//         return;
//     }

        
//     ofstream outfile(outpath, ios::binary | ios::app);
//     if (!outfile) {
//         cerr << "Error: Could not open outfile for writing.\n";
//         return;
//     }

//     ListmodeRecord rec;
//     while (infile.read(reinterpret_cast<char*>(&rec), sizeof(ListmodeRecord))) {
//         outfile.write(reinterpret_cast<char*>(&rec), sizeof(ListmodeRecord));
//     }
// }


void combine_lm(vector<string> inpaths, string outpath) {

    ifstream ins[LORS];
    int total_recs = 0;
    int recs[LORS];

    random_device rd;  // Will be used to obtain a seed for the random number engine
    mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
    uniform_real_distribution<double> dis(0.0, 1.0);

    for (int i = 0; i < LORS; i++) {
        ins[i] = ifstream(inpaths[i], ios::binary);
        if (!ins[i]) {
            cerr << "Error: Could not open infile for reading. \n";
            return;
        }
        if (filesystem::file_size(inpaths[i]) % sizeof(ListmodeRecord) != 0) {
            cerr << "Error: file not sized correctly.\n";
            return;
        }
        int records_here = filesystem::file_size(inpaths[i]) / sizeof(ListmodeRecord);
        recs[i] = records_here;
        total_recs += recs[i];
    }
        
    ofstream outfile(outpath, ios::binary | ios::app);
    if (!outfile) {
        cerr << "Error: Could not open outfile for writing.\n";
        return;
    }

    ListmodeRecord rec;

    while(total_recs > 0) {
        if (total_recs % 1000000 == 0) {
            cout << "remaining: " << total_recs << endl;
        }
        int index = dis(gen) * total_recs;
        int i = 0;
        int sum = 0;
        for (; i < LORS; ++i) {
            if (sum + recs[i] > index) {
                break;
            }
            sum += recs[i];
        }

        if (recs[i] == 0) {
            cout << "Error with choosing file." << endl;
        }

        recs[i]--;
        total_recs--;

        ins[i].read(reinterpret_cast<char*>(&rec), sizeof(ListmodeRecord));
        outfile.write(reinterpret_cast<char*>(&rec), sizeof(ListmodeRecord));
    }
}


void test_source(string inpath, double x, double y, double z) {
    // Open Infile
    ifstream file(inpath, ios::binary);
    if (!file) {
        cerr << "Error: Could not open file.\n";
        return;
    }

    Record rec;
    while (read_record(file, rec)) {
        if (abs(rec.srcX - x) < 1 && abs(rec.srcY - y) < 1 && abs(rec.srcZ - z) < 1) {
            cout << "Src: " << rec.srcX << " " << rec.srcY << " " << rec.srcZ << endl;
            break;
        }
    }
}


// void shuffle_lm(string inpath, string outpath) {
//     py::object np = py::module_::import("numpy");

//     ifstream infile(inpath, ios::binary);
//     if (!infile) {
//         cerr << "Error: Could not open infile for reading.\n";
//         return;
//     }

//     ofstream outfile(outpath, ios::binary | ios::app);
//     if (!outfile) {
//         cerr << "Error: Could not open outfile for writing.\n";
//         return;
//     }

//     int numrecords = filesystem::file_size(inpath) / sizeof(ListmodeRecord);
//     if (filesystem::file_size(inpath) % sizeof(ListmodeRecord) != 0) {
//         cerr << "Error: file size not multiple of record size. \n";
//         return;
//     }

//     // write_indices(numrecords);

//     py::array_t<int> pos = np.attr("memmap")("temp.npy", np.attr("int32"));
//     auto pos_buf = pos.request();
//     int *pos_ptr = (int*) pos_buf.ptr;

//     ifstream indices(inpath, ios::binary);
//     if (!indices) {
//         cerr << "Error: Could not open indices file for reading.\n";
//         return;
//     }

//     py::array_t<float> in = np.attr("memmap")(inpath, np.attr("float32"));
//     auto in_buf = in.request();
//     float *in_ptr = (float*) in_buf.ptr;
    
//     int npos;
//     for (int i = 0; i < numrecords; i++) {
//         if (i % 10000000 == 0) {
//             cout << "Read " << i << "records" << endl;
//         }

//         npos = pos_ptr[i];

        

//         outfile.write(reinterpret_cast<char*>(&rec), sizeof(ListmodeRecord));
//     }
// }


PYBIND11_MODULE(randoms, m) {
    m.def("bundle_coincidences", &bundle_coincidences, "Bundles coincidences",
        py::arg("times"),
        py::arg("TAU")
    );
    m.def("singles_counts", &singles_counts, "calculates singles counts per detector",
        py::arg("detector_occurances"),
        py::arg("num_detectors")
    );
    m.def("prompts_counts", &prompts_counts, "calculates prompts counts per detector",
        py::arg("det1_hits"),
        py::arg("det2_hits"),
        py::arg("num_detectors")
    );
    m.def("coincidences_per_lor", &coincidences_per_lor, "calculates coincidences per LOR",
        py::arg("det1_hits"),
        py::arg("det2_hits"),
        py::arg("num_detectors")
    );
    m.def("sp_rates", &sp_rates, "Calculates sp rates",
        py::arg("singles_count"),
        py::arg("prompts_count"),
        py::arg("num_detectors"),
        py::arg("L"),
        py::arg("S"),
        py::arg("TAU"),
        py::arg("TIME")
    );
    m.def("sp_correction", &sp_correction, "Calculates sp multi-correction terms",
        py::arg("singles_count"),
        py::arg("num_detectors"),
        py::arg("exp_prod"),
        py::arg("TAU"),
        py::arg("TIME")
    );
    m.def("dw_rates", &dw_rates, "calculates dw estimate", 
        py::arg("times"),
        py::arg("detectors"),
        py::arg("num_detectors"),
        py::arg("TAU"),
        py::arg("DELAY")
    );
    m.def("sr_rates", &sr_rates, "calculates sr estimate",
        py::arg("singles_count"),
        py::arg("num_detectors"),
        py::arg("TAU"),
        py::arg("TIME")
    );
    m.def("get_times", &get_times, "gives raw time from coarse and fine",
        py::arg("coarse"),
        py::arg("fine"),
        py::arg("coarse_t"),
        py::arg("fine_t")
    );
    m.def("read_file", &read_file, "reads file",
        py::arg("path"),
        py::arg("TAU"),
        py::arg("DELAY"),
        py::arg("num_detectors")
    );
    m.def("read_file_lm", &read_file_lm, "reads file and writes listmode",
        py::arg("path"),
        py::arg("outfolder"),
        py::arg("name"),
        py::arg("TAU"),
        py::arg("DELAY"),
        py::arg("num_detectors")
    );
    m.def("hist_tof", &hist_tof, "generates hist of TOF",
        py::arg("path"),
        py::arg("abs_max"),
        py::arg("num_bins")
    );
    m.def("split_lm", &split_lm, "splits lm by LORs",
        py::arg("inpath"),
        py::arg("outpath"),
        py::arg("name"),
        py::arg("detectors")
    );
    m.def("combine_lm", &combine_lm, "combines listmode files",
        py::arg("inpaths"),
        py::arg("outpath")
    );
    m.def("test_source", &test_source, "tests source pos",
        py::arg("inpath"),
        py::arg("x"),
        py::arg("y"),
        py::arg("z")
    );
    // m.def("shuffle_lm", &shuffle_lm, "shuffles listmode files",
    //     py::arg("inpath"),
    //     py::arg("outpath")
    // );
}