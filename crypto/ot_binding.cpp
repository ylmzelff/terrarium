#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "priority_ot.h"

namespace py = pybind11;

// Convert mpz_class to Python int
py::int_ mpz_to_pyint(const mpz_class& value) {
    return py::cast(PyLong_FromString(value.get_str().c_str(), nullptr, 10));
}

// Convert Python int to mpz_class
mpz_class pyint_to_mpz(const py::int_& value) {
    return mpz_class(py::str(py::cast<py::object>(value)).cast<std::string>());
}

// Wrapper functions with Python-friendly types
std::vector<std::vector<std::string>> Setup_wrapper(int number_of_OT, int n, unsigned int bit_size) {
    auto result = Setup(number_of_OT, n, bit_size);
    std::vector<std::vector<std::string>> py_result;
    for (const auto& vec : result) {
        std::vector<std::string> py_vec;
        for (const auto& val : vec) {
            py_vec.push_back(val.get_str());
        }
        py_result.push_back(py_vec);
    }
    return py_result;
}

std::vector<std::vector<std::string>> GenRes_wrapper(
    const std::vector<std::string>& m_str,
    int number_of_OT,
    const std::vector<std::vector<std::string>>& r_str,
    const std::vector<std::vector<int>>& w) {
    
    // Convert string inputs to mpz_class
    std::vector<mpz_class> m;
    for (const auto& s : m_str) {
        m.push_back(mpz_class(s));
    }
    
    std::vector<std::vector<mpz_class>> r;
    for (const auto& vec : r_str) {
        std::vector<mpz_class> r_vec;
        for (const auto& s : vec) {
            r_vec.push_back(mpz_class(s));
        }
        r.push_back(r_vec);
    }
    
    // Call C++ function
    auto result = GenRes(m, number_of_OT, r, w);
    
    // Convert back to strings
    std::vector<std::vector<std::string>> py_result;
    for (const auto& vec : result) {
        std::vector<std::string> py_vec;
        for (const auto& val : vec) {
            py_vec.push_back(val.get_str());
        }
        py_result.push_back(py_vec);
    }
    return py_result;
}

std::vector<std::vector<std::string>> oblFilter_wrapper(
    int number_of_OT,
    int p_size,
    const std::vector<std::vector<std::string>>& res_s_str,
    const std::vector<std::vector<int>>& y) {
    
    // Convert strings to mpz_class
    std::vector<std::vector<mpz_class>> res_s;
    for (const auto& vec : res_s_str) {
        std::vector<mpz_class> res_vec;
        for (const auto& s : vec) {
            res_vec.push_back(mpz_class(s));
        }
        res_s.push_back(res_vec);
    }
    
    auto result = oblFilter(number_of_OT, p_size, res_s, y);
    
    // Convert back to strings
    std::vector<std::vector<std::string>> py_result;
    for (const auto& vec : result) {
        std::vector<std::string> py_vec;
        for (const auto& val : vec) {
            py_vec.push_back(val.get_str());
        }
        py_result.push_back(py_vec);
    }
    return py_result;
}

std::string retreive_wrapper(
    const std::string& res_h_str,
    int j,
    const std::vector<std::string>& r_str,
    const std::vector<int>& p) {
    
    mpz_class res_h(res_h_str);
    std::vector<mpz_class> r;
    for (const auto& s : r_str) {
        r.push_back(mpz_class(s));
    }
    
    auto result = retreive(res_h, j, r, p);
    return result.get_str();
}

PYBIND11_MODULE(pyot, m) {
    m.doc() = "Priority Oblivious Transfer protocol for privacy-preserving slot intersection";
    
    m.def("setup", &Setup_wrapper, 
          "Phase 1: Generate random encryption keys",
          py::arg("number_of_OT"), py::arg("n"), py::arg("bit_size"));
    
    m.def("gen_query", &genQuery,
          "Phase 2: Generate query and permutation vectors",
          py::arg("number_of_OT"), py::arg("p_size"), py::arg("p"), py::arg("n"), py::arg("y"));
    
    m.def("gen_res", &GenRes_wrapper,
          "Phase 3: Generate encrypted response",
          py::arg("m"), py::arg("number_of_OT"), py::arg("r"), py::arg("w"));
    
    m.def("obl_filter", &oblFilter_wrapper,
          "Phase 4: Oblivious filtering",
          py::arg("number_of_OT"), py::arg("p_size"), py::arg("res_s"), py::arg("y"));
    
    m.def("retreive", &retreive_wrapper,
          "Phase 5: Retrieve and decrypt message",
          py::arg("res_h"), py::arg("j"), py::arg("r"), py::arg("p"));
}
