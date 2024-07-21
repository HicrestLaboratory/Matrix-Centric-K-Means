#include <iostream>
#include <vector>
#include <chrono>
#include <random>

#include <dlib/clustering.h>
#include <dlib/rand.h>

#include "ArgParse.hpp"

using namespace dlib;

using IT = uint32_t;
using DT = float;
using point_t =  matrix<DT, 0, 1> ;
using kernel_t = linear_kernel<point_t>;

template <typename T>
void read_svm(const uint32_t n, const uint32_t d,
                const std::string& infile, T * h_dataset)
{
    std::ifstream in;
    in.open(infile);

    std::string str;

    int i = 0;
    while (std::getline(in, str, '\n')) {
        std::istringstream input_str(str);
        std::string token;
        while (std::getline(input_str, token, ' ')) {
            std::istringstream token_stream(token);
            std::string key, value;
            if (std::getline(token_stream, key, ':') &&
                std::getline(token_stream, value)) {
                    h_dataset[std::atoi(key.c_str())-1 + d*i] = std::atof(value.c_str());
            }
        }
        i++;
    }

    in.close();

}


template <typename T>
void read_csv(const uint32_t n, const uint32_t d,
				const std::string& infile, T * h_dataset)
{
    std::ifstream in;
    in.open(infile);

    std::string str;

    int i = 0;
    while (std::getline(in, str, '\n')) {

        /* Skip header */
        if (i==0) {
            i++;
            continue;
        }

        std::istringstream input_str(str);
        std::string token;
        int j = 0;
        while (std::getline(input_str, token, ',')) {

            /* Skip class label */
            if (j==0) {
                j++;
                continue;
            }

            std::istringstream token_stream(token);
            h_dataset[j-1 + d*(i-1)] = std::atof(token.c_str());
            j++;
        }
        i++;
    }
    
    in.close();
}


template <typename T>
void init_rand(const IT n, const IT d, T * dataset)
{
    std::random_device rd;
    std::mt19937 gen(11);
    std::uniform_real_distribution<T> distr(-1e5, 1e5);

    for (int i=0; i<n; i++) {
        for (int j=0; j<d; j++) {
            dataset[j + d*i] = distr(gen); 
        }
    }
}


template <typename Kmeans>
double compute_inertia(const Kmeans& kmeans,
                       const std::vector<point_t>& samples, 
                       kernel_t& kernel) 
{
    double inertia = 0.0;

    for (size_t i = 0; i < samples.size(); i++) {

        //DT best_score = kernel(kmeans(samples[i]), kmeans(samples[i])) + kernel(samples[i], samples[i]) - 2*kernel(samples[i], kmeans(samples[i]));
        auto cluster = kmeans(samples[i]);
        auto center = kmeans.get_kcentroid(cluster);
        DT best_score = (center)(samples[i]);

        inertia += best_score*best_score ;

    }

    return inertia;
}


struct Params {
    uint32_t maxiters;
    uint32_t ntrials;
    float tol;
    std::string infile;
};


void run_kkmeans(const IT n,
                 const IT d,
                 const IT k,
                 Params * params)
{

    kernel_t kernel = kernel_t( );

    kcentroid<kernel_t> kc(kernel, params->tol);

    kkmeans<kernel_t> kmeans(kc);


    point_t point;
    point.set_size(d);
    
    std::vector<point_t> points;
    points.reserve(n);

    DT * dataset = new DT[n*d];

    if (params->infile.size()==0) {
        init_rand(n, d, dataset);
    } else if (params->infile.find(".csv")!=std::string::npos) {
        std::cout<<"csv"<<std::endl;
        read_csv(n, d, params->infile, dataset);
    } else if (params->infile.find(".svm")!=std::string::npos) {
        read_svm(n, d, params->infile, dataset);
    } else {
        std::cerr<<params->infile<<" is invalid infile"<<std::endl;
        exit(1);
    }


    for (int i=0; i<n; i++) {
        for (int j=0; j<d; j++) {
            point(j) = dataset[j + i*d];
        }
        points.push_back(point);
    }


    kmeans.set_number_of_centers(k);

    std::vector<point_t> centers(k);

    std::random_device rd;
    std::mt19937 gen(11);
    std::uniform_int_distribution<> distr(0, n);

    for (int i=0; i<k; i++) {
        centers[i] = points[distr(gen)];
    }

    double kmeans_time = 0;
    double kmeans_score = 0;

    for (int i=0; i<params->ntrials; i++) {

        std::cerr<<"Trial "<<i<<std::endl;

        auto stime = std::chrono::system_clock::now();
        kmeans.train(points, centers, params->maxiters);
        auto etime = std::chrono::system_clock::now();

        auto kmeans_duration = std::chrono::duration_cast<std::chrono::duration<double>>(etime - stime);

        double score = compute_inertia(kmeans, points, kernel);
        kmeans_score += score;


        if (i > 0) {
            kmeans_time += kmeans_duration.count();
        }

    }

    kmeans_time /= (params->ntrials - 1);
    kmeans_score /= (params->ntrials);

    std::cout<<"kmeans-time: "<<kmeans_time<<"s"<<std::endl;
    std::cout<<"kmeans-score: "<<kmeans_score<<std::endl;





    delete[] dataset;
}


int main(int argc, char ** argv)
{
    ArgParse args(argc, argv);

    IT n = args.getArgInt<IT>("n");
    IT d = args.getArgInt<IT>("d");
    IT k = args.getArgInt<IT>("k");

    if (!args.isArgPresent("infile"))
        args.arguments_["infile"] = "";

    Params * params = new Params{args.getArgInt<uint32_t>("maxiters"),
                                 args.getArgInt<uint32_t>("ntrials"),
                                 args.getArgFloat("tol"),
                                 args.getArg("infile")};

    run_kkmeans(n, d, k, params);

    delete params;

	
    return 0;
}



