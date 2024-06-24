#ifndef __INPUT_PARSER__
#define __INPUT_PARSER__

#include <string.h>
#include "point.hpp"
#include <random>

#define SEPARATOR ","
#define MAX_LINE   8192
#define PRECISION  8

using namespace std;

template <typename T>
class InputParser {
  private:
    Point<T> **dataset;
    int d;
    size_t n;

  public:
    //NOTE: This has been modified to work with the libsvm data format
    InputParser (istream &in, int _d, size_t _n) {
      this->dataset = new Point<T>*[_n];
      this->n = _n;
      this->d = _d;

      std::string str;

      T *point = new T[_d];

      int i = 0;
      while (std::getline(in, str, '\n')) {
        std::istringstream input_str(str);
        std::string token;
        while (std::getline(input_str, token, ' ')) {
            std::istringstream token_stream(token);
            std::string key, value;
            if (std::getline(token_stream, key, ':') &&
                std::getline(token_stream, value)) {
                    point[std::atoi(key.c_str())-1] = std::atof(value.c_str());
            }
        }
        dataset[i] = new Point<T>(point, _d);
        i++;
        memset(point, 0, sizeof(T)*d);
      }

      delete[] point;
    }


    /* Generate random data according to seed */
    InputParser(int seed, int _d, size_t _n):
        n(_n), d(_d), dataset(new Point<T>*[_n])
    {
        
        std::mt19937 eng(seed);
        std::uniform_real_distribution<float> distr(-1e5, 1e5);

        T * point = new T[d];
        for (size_t i=0; i<n; i++) {

            for (int j=0; j<d; j++) {
                point[j] = distr(eng);
            }
            dataset[i] = new Point<T>(point, d);
            //memset(point, 0, sizeof(T)*d);
        }

        delete[] point;
    }

    ~InputParser() {
      for (size_t i = 0; i < n; ++i) delete dataset[i];
      delete[] dataset;
    }

    Point<T> **get_dataset () { return dataset; }
    size_t get_dataset_size() { return n; };

    void dataset_to_csv(ostream& o) {
      o << "cluster" << SEPARATOR;
      for (int i = 0; i < d; ++i) {
        o << "d" << i;
        if (i != (d - 1)) o << SEPARATOR;
      }
      o << endl;

      for (size_t i = 0; i < n; ++i) {
        o << dataset[i]->getCluster() << SEPARATOR;
        for (int j = 0; j < d; ++j) {
          o << setprecision(PRECISION) << dataset[i]->get(j);
          if (j != (d - 1)) o << SEPARATOR;
        }
        o << endl;
      }
    }

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
