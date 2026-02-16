#ifndef PRIORITY_OT_H
#define PRIORITY_OT_H

#include <vector>
#include <gmpxx.h>
#include <unordered_map>

using namespace std;

// Phase 1: Setup - Generate random encryption keys
vector<vector<mpz_class>> Setup(int number_of_OT, int n, unsigned int bit_size);

// Phase 2: Query Generation
vector<vector<int>> genQuery(int number_of_OT, int p_size, vector<vector<int>> p, int n, vector<vector<int>>& y);

// Phase 3: Generate Response
vector<vector<mpz_class>> GenRes(const vector<mpz_class>& m, int number_of_OT, 
                                  const vector<vector<mpz_class>>& r, const vector<vector<int>>& w);

// Phase 4: Oblivious Filter
vector<vector<mpz_class>> oblFilter(int number_of_OT, int p_size, 
                                     const vector<vector<mpz_class>>& res_s, const vector<vector<int>>& y);

// Phase 5: Retrieve
mpz_class retreive(const mpz_class& res_h, int j, const vector<mpz_class>& r, const vector<int>& p);

// Helper function
unordered_map<int, int> createIndexMap(const vector<int>& v);

#endif // PRIORITY_OT_H
