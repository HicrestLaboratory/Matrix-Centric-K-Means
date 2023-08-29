#ifndef __ERRORS__
#define __ERRORS__

#include <iostream>
#include <string>
#include <fstream>

#include "cxxopts.hpp"
#include "input_parser.hpp"

#ifndef DATA_TYPE
  #define DATA_TYPE float
#endif

#define ARG_DIM         "n-dimensions"
#define ARG_SAMPLES     "n-samples"
#define ARG_CLUSTERS    "n-clusters"
#define ARG_MAXITER     "maxiter"
#define ARG_OUTFILE     "out-file"
#define ARG_INFILE      "in-file"
#define ARG_TOL         "tolerance"
#define ARG_RUNS        "runs"
#define ARG_SEED        "seed"

const char* ARG_STR[]   = {"dimensions", "n-samples", "clusters", "maxiter", "out-file", "in-file", "tolerance"};
const float DEF_EPSILON = numeric_limits<float>::epsilon();
const int   DEF_RUNS    = 1;

#define EXIT_ARGS             1
#define EXIT_CUDA_DEV         2
#define EXIT_POINT_IOB        3
#define EXIT_INVALID_INFILE   4

void printErrDesc (int errn) {
  switch (errn) {
  case EXIT_ARGS:
    cerr << "Invalid or missing argument: ";
    break;
  case EXIT_CUDA_DEV:
    cerr << "There are no available device(s) that support CUDA" << endl;
    break;
  case EXIT_POINT_IOB:
    cerr << "Point index out of bounds" << endl;
    break;
  case EXIT_INVALID_INFILE:
    cerr << "Invalid input file" << endl;
    break;
  default:
    cerr << "No error description" << endl;
    break;
  }
}

int getArg_u (const cxxopts::ParseResult &args, const char *arg, const int *def_val) {
  try {
    return args[arg].as<int>();
  } catch(...) {
    if (def_val) { return *def_val; }
    printErrDesc(EXIT_ARGS);
    cerr << arg << endl;
    exit(EXIT_ARGS);
  }
}

float getArg_f (const cxxopts::ParseResult &args, const char *arg, const float *def_val) {
  try {
    return args[arg].as<float>();
  } catch(...) {
    if (def_val) { return *def_val; }
    printErrDesc(EXIT_ARGS);
    cerr << arg << endl;
    exit(EXIT_ARGS);
  }
}

string getArg_s (const cxxopts::ParseResult &args, const char *arg, const string *def_val) {
  try {
    return args[arg].as<string>();
  } catch(...) {
    if (def_val) { return *def_val; }
    printErrDesc(EXIT_ARGS);
    cerr << arg << endl;
    exit(EXIT_ARGS);
  }
}

void parse_input_args(const int argc, const char *const *argv, uint32_t *d, size_t *n, uint32_t *k, size_t *maxiter, string &out_file, float *tol, uint32_t *runs, int **seed, InputParser<DATA_TYPE> **input) {
  cxxopts::Options options("gpukmeans", "gpukmeans is an implementation of the K-means algorithm that uses a GPU");

  options.add_options()
    ("h,help", "Print usage")
    ("d," ARG_DIM,      "Number of dimensions of a point",  cxxopts::value<int>())
    ("n," ARG_SAMPLES,  "Number of points",                 cxxopts::value<int>())
    ("k," ARG_CLUSTERS, "Number of clusters",               cxxopts::value<int>())
    ("m," ARG_MAXITER,  "Maximum number of iterations",     cxxopts::value<int>())
    ("o," ARG_OUTFILE,  "Output filename",                  cxxopts::value<string>())
    ("i," ARG_INFILE,   "Input filename",                   cxxopts::value<string>())
    ("r," ARG_RUNS,     "Number of k-means runs",           cxxopts::value<int>()->default_value(to_string(DEF_RUNS)))
    ("s," ARG_SEED,     "Seed for centroids generator",     cxxopts::value<int>())
    ("t," ARG_TOL,      "Tolerance to declare convergence", cxxopts::value<float>()->default_value(to_string(DEF_EPSILON)));

  cxxopts::ParseResult args = options.parse(argc, argv);

  if (args.count("help")) {
    cout << options.help() << endl;
    exit(0);
  }

  *d        = getArg_u(args, ARG_DIM,      NULL);
  *n        = getArg_u(args, ARG_SAMPLES,  NULL);
  *k        = getArg_u(args, ARG_CLUSTERS, NULL);
  *maxiter  = getArg_u(args, ARG_MAXITER,  NULL);
  out_file  = getArg_s(args, ARG_OUTFILE,  NULL);
  *tol      = getArg_f(args, ARG_TOL,      &DEF_EPSILON);
  *runs     = getArg_u(args, ARG_RUNS,     &DEF_RUNS);

  *seed = NULL;
  if (args[ARG_SEED].count() > 0) {
    int in_seed = getArg_u(args, ARG_SEED, NULL);
    *seed = new int(in_seed);
  }

  if(args[ARG_INFILE].count() > 0) {
    const string in_file = getArg_s(args, ARG_INFILE, NULL);
    filebuf fb;
    if (fb.open(in_file, ios::in)) {
      istream file(&fb);
      *input = new InputParser<DATA_TYPE>(file, *d, *n);
      fb.close();
    } else {
      printErrDesc(EXIT_INVALID_INFILE);
      exit(EXIT_INVALID_INFILE);
    }
  } else {
    *input = new InputParser<DATA_TYPE>(cin, *d, *n);
  }
}

#endif