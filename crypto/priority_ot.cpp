// Author: Aydin Abadi - aydin.abadi@ncl.ac.uk
// Adapted for Terrarium: Privacy-preserving slot intersection via Priority OT

#include "priority_ot.h"
#include <algorithm>
#include <random>
#include <chrono>
#include <iostream>

using namespace std;

// Phase 1: Setup - Generate random encryption keys
vector<vector<mpz_class>> Setup(int number_of_OT, int n, unsigned int bit_size) {
    vector<vector<mpz_class>> random_numbers_collection;
    random_numbers_collection.reserve(number_of_OT);
    
    gmp_randclass rand_gen(gmp_randinit_default);
    rand_gen.seed(time(nullptr));
    
    for (int j = 0; j < number_of_OT; ++j) {
        vector<mpz_class> v(n);
        for (int i = 0; i < n; ++i) {
            v[i] = rand_gen.get_z_bits(bit_size);
        }
        random_numbers_collection.push_back(move(v));
    }
    return random_numbers_collection;
}

// Helper: Create index map for fast lookup
unordered_map<int, int> createIndexMap(const vector<int>& v) {
    unordered_map<int, int> indexMap;
    // DÜZELTME: i için size_t kullanılarak v.size() ile tip uyumu sağlandı.
    for (size_t i = 0; i < v.size(); ++i) {
        indexMap[v[i]] = (int)i;
    }
    return indexMap;
}

// Phase 2: Query Generation - Generate permutation and query vectors
vector<vector<int>> genQuery(int number_of_OT, int p_size, vector<vector<int>> p, int n, vector<vector<int>>& y) {
    vector<vector<int>> result(number_of_OT);
    y.resize(number_of_OT);
    int t = p_size;
    
    vector<int> v(n);
    for (int i = 0; i < n; ++i) {
        v[i] = i;
    }
    
    random_device rd;
    default_random_engine rng(rd());
    
    for (int i = 0; i < number_of_OT; ++i) {
        shuffle(v.begin(), v.end(), rng);
        result[i] = v;
        y[i].resize(t);
        
        if (n == 2) {
            for (int j = 0; j < t; ++j) {
                y[i][j] = (p[i][j] == 0) ? v[0] : v[1];
            }
        } else {
            unordered_map<int, int> map;
            map.reserve(n);
            // DÜZELTME: j için size_t kullanıldı.
            for (size_t j = 0; j < (size_t)n; ++j) {
                map[v[j]] = (int)j;
            }
            for (int j = 0; j < t; ++j) {
                y[i][j] = map[p[i][j]];
            }
        }
    }
    return result;
}

// Phase 3: Generate Response - Encrypt and permute messages
vector<vector<mpz_class>> GenRes(const vector<mpz_class>& m, int number_of_OT, 
                                  const vector<vector<mpz_class>>& r, const vector<vector<int>>& w) {
    size_t m_size = m.size();
    vector<vector<mpz_class>> x(number_of_OT, vector<mpz_class>(m_size));
    
    for (int j = 0; j < number_of_OT; ++j) {
        unordered_map<int, int> indexMap = createIndexMap(w[j]);
        for (size_t i = 0; i < m_size; ++i) {
            mpz_class xor_result = m[i] ^ r[j][i];
            auto it = indexMap.find((int)i);
            if (it != indexMap.end()) {
                int y_i = it->second;
                x[j][y_i] = xor_result;
            } else {
                cerr << "Error: invalid index during GenRes" << endl;
                return {};
            }
        }
    }
    return x;
}

// Phase 4: Oblivious Filter - Filter encrypted messages
vector<vector<mpz_class>> oblFilter(int number_of_OT, int p_size, 
                                       const vector<vector<mpz_class>>& res_s, const vector<vector<int>>& y) {
    vector<vector<mpz_class>> res(number_of_OT, vector<mpz_class>(p_size));
    for (int j = 0; j < number_of_OT; ++j) {
        for (int i = 0; i < p_size; ++i) {
            res[j][i] = res_s[j][y[j][i]];
        }
    }
    return res;
}

// Phase 5: Retrieve - Decrypt message
mpz_class retreive(const mpz_class& res_h, int j, const vector<mpz_class>& r, const vector<int>& p) {
    // DÜZELTME: j'nin p.size() ile güvenli karşılaştırması için casting yapıldı.
    if (j < 0 || (size_t)j >= p.size()) {
        throw runtime_error("Priority index out of range");
    }
    return res_h ^ r[p[j]];
}