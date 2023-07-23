#ifndef __INPUT_PARSER__
#define __INPUT_PARSER__

#include <string.h>
#include "point.hpp"

#define SEPARATOR ","
#define MAX_LINE  8192

using namespace std;

template <typename T> 
class InputParser {
  private:
    Point<T> **dataset;
    int d;
    size_t n;

  public:
    InputParser (istream &in, int _d, size_t _n) {
      this->dataset = new Point<T>*[_n];
      this->n = _n;
      this->d = _d;
      
      T *point = new T[_d];
      for (size_t i = 0; i <= _n; i++) {
        char str[MAX_LINE] = { 0 };
        in >> str;
        // printf("%s,\n", str);
        
        if (i == (size_t)0) { continue; }
        if (!str[0]) { break; }
        
        int j = 0;
        char *tok = strtok(str, SEPARATOR);
        while (tok && j < _d) {
          point[j++] = atof(tok);
          tok = strtok(NULL, SEPARATOR);
        }

        dataset[i - 1] = new Point<T>(point, _d);
      }
      
      delete[] point;
    }

    ~InputParser() {
      for (size_t i = 0; i < n; ++i) delete dataset[i];
      delete[] dataset;
    }

    Point<T> **get_dataset () { return dataset; }
    size_t get_dataset_size() { return n; };

    friend ostream& operator<< (ostream &os, InputParser const& p) {
      const int W = 9;
      os << "   i  cluster";
      for (int i = 0; i < p.d; ++i) {
        char s[W];
        sprintf(s, "d%d", i);
        os << setw(W) << s;
      }
      os << endl;
      for (size_t i = 0; i < min(p.n, (size_t)5); ++i) {
        os << setw(4) << i << setw(W) << *p.dataset[i] << endl;
      }
      if (p.n > 5) {
        os << " ..." << endl; 
        for (size_t i = p.n - 5; i < p.n; ++i) {
          os << setw(4) << i << setw(W) << *p.dataset[i] << endl;
        }
      }
      os << endl << "[" << p.n << " rows x " << p.d << " columns]" << endl;
      return os;
    }
};

#endif