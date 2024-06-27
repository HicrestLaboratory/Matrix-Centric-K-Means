
#include "common.cuh"

#include <raft/core/device_mdarray.hpp>
#include <raft/cluster/kmeans.cuh>
#include <raft/cluster/kmeans_types.hpp>
#include <raft/core/resources.hpp>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>

#include <ctime>




#include <cstdint>
#include <optional>

#define SEPARATOR ","
#define MAX_LINE   8192

int n_trials = 10;

namespace raft {
namespace cluster {
namespace detail {
template <typename DataT, typename IndexT>
void my_kmeans_fit_main(raft::resources const& handle,
                                         const cluster::KMeansParams& params,
                                         raft::device_matrix_view<const DataT, IndexT> X,
                                         raft::device_vector_view<const DataT, IndexT> weight,
                                         raft::device_matrix_view<DataT, IndexT> centroidsRawData,
                                         raft::host_scalar_view<DataT> inertia,
                                         raft::host_scalar_view<IndexT> n_iter,
                                         rmm::device_uvector<char>& workspace,
                                         bool check_converged=false)
{
    common::nvtx::range<common::nvtx::domain::raft> fun_scope("kmeans_fit_main");
    logger::get(RAFT_NAME).set_level(params.verbosity);
    cudaStream_t stream = resource::get_cuda_stream(handle);
    auto n_samples          = X.extent(0);
    auto n_features         = X.extent(1);
    auto n_clusters         = params.n_clusters;
    auto metric                 = params.metric;

    // stores (key, value) pair corresponding to each sample where
    //   - key is the index of nearest cluster
    //   - value is the distance to the nearest cluster
    auto minClusterAndDistance =
        raft::make_device_vector<raft::KeyValuePair<IndexT, DataT>, IndexT>(handle, n_samples);

    // temporary buffer to store L2 norm of centroids or distance matrix,
    // destructor releases the resource
    rmm::device_uvector<DataT> L2NormBuf_OR_DistBuf(0, stream);

    // temporary buffer to store intermediate centroids, destructor releases the
    // resource
    auto newCentroids = raft::make_device_matrix<DataT, IndexT>(handle, n_clusters, n_features);

    // temporary buffer to store weights per cluster, destructor releases the
    // resource
    auto wtInCluster = raft::make_device_vector<DataT, IndexT>(handle, n_clusters);

    rmm::device_scalar<DataT> clusterCostD(stream);

    // L2 norm of X: ||x||^2
    auto L2NormX = raft::make_device_vector<DataT, IndexT>(handle, n_samples);
    auto l2normx_view =
        raft::make_device_vector_view<const DataT, IndexT>(L2NormX.data_handle(), n_samples);

    if (metric == raft::distance::DistanceType::L2Expanded ||
            metric == raft::distance::DistanceType::L2SqrtExpanded) {
        raft::linalg::rowNorm(L2NormX.data_handle(),
                                                    X.data_handle(),
                                                    X.extent(1),
                                                    X.extent(0),
                                                    raft::linalg::L2Norm,
                                                    true,
                                                    stream);
    }

    RAFT_LOG_DEBUG(
        "Calling KMeans.fit with %d samples of input data and the initialized "
        "cluster centers",
        n_samples);

    double update_time = 0.0;
    double dist_time = 0.0;
    IndexT d = X.extent(1);
    IndexT n = X.extent(0);
    IndexT k = n_clusters;

    DataT priorClusteringCost = 0;
#if LOG
    std::ofstream centroids_out;
    centroids_out.open("centroids-raft.out");
#endif
    for (n_iter[0] = 1; n_iter[0] <= params.max_iter; ++n_iter[0]) {
        RAFT_LOG_DEBUG(
            "KMeans.fit: Iteration-%d: fitting the model using the initialized "
            "cluster centers",
            n_iter[0]);
        //std::cout<<"iter "<<n_iter[0]<<std::endl;
#if LOG

        DataT * h_centroids = new DataT[d*k];
        raft::copy(h_centroids, centroidsRawData.data_handle(), k*d, stream);

        for (int i=0; i<k; i++) {
            for (int j=0; j<d; j++) {
                centroids_out<<h_centroids[d*i + j]<<",";
            }
            centroids_out<<std::endl;
        }


        delete[] h_centroids;
#endif

        auto centroids = raft::make_device_matrix_view<DataT, IndexT>(
            centroidsRawData.data_handle(), n_clusters, n_features);

        // computes minClusterAndDistance[0:n_samples) where
        // minClusterAndDistance[i] is a <key, value> pair where
        //   'key' is index to a sample in 'centroids' (index of the nearest
        //   centroid) and 'value' is the distance between the sample 'X[i]' and the
        //   'centroid[key]'
        auto stime_dist = std::chrono::system_clock::now();
        detail::minClusterAndDistanceCompute<DataT, IndexT>(handle,
                                                            X,
                                                            centroids,
                                                            minClusterAndDistance.view(),
                                                            l2normx_view,
                                                            L2NormBuf_OR_DistBuf,
                                                            params.metric,
                                                            params.batch_samples,
                                                            params.batch_centroids,
                                                            workspace);
        auto etime_dist = std::chrono::system_clock::now();
        auto dist_duration = std::chrono::duration_cast<std::chrono::duration<double>>(etime_dist-stime_dist);
        dist_time += dist_duration.count();

        // Using TransformInputIteratorT to dereference an array of
        // raft::KeyValuePair and converting them to just return the Key to be used
        // in reduce_rows_by_key prims
        detail::KeyValueIndexOp<IndexT, DataT> conversion_op;
        cub::TransformInputIterator<IndexT,
                                                                detail::KeyValueIndexOp<IndexT, DataT>,
                                                                raft::KeyValuePair<IndexT, DataT>*>
            itr(minClusterAndDistance.data_handle(), conversion_op);
#if LOG
        thrust::device_vector<uint32_t> d_clusters(n);
        thrust::copy(itr, itr+n, d_clusters.begin());
        uint32_t * h_clusters = new uint32_t[n];
        cudaMemcpy(h_clusters, thrust::raw_pointer_cast(d_clusters.data()), sizeof(uint32_t)*n, cudaMemcpyDeviceToHost);
        centroids_out<<"CLUSTERS"<<std::endl;
        for (int i=0; i<n; i++) {
            centroids_out<<h_clusters[i]<<",";
        }
        centroids_out<<std::endl;
        delete[] h_clusters;
#endif

        auto stime_update = std::chrono::system_clock::now();
        update_centroids(handle,
                                 X,
                                 weight,
                                 raft::make_device_matrix_view<const DataT, IndexT>(
                                     centroidsRawData.data_handle(), n_clusters, n_features),
                                 itr,
                                 wtInCluster.view(),
                                 newCentroids.view(),
                                 workspace);
        auto etime_update = std::chrono::system_clock::now();
        auto update_duration = std::chrono::duration_cast<std::chrono::duration<double>>(etime_update-stime_update);
        update_time += update_duration.count();

        // compute the squared norm between the newCentroids and the original
        // centroids, destructor releases the resource
        auto sqrdNorm = raft::make_device_scalar(handle, DataT(0));
        raft::linalg::mapThenSumReduce(sqrdNorm.data_handle(),
                                                                     newCentroids.size(),
                                                                     raft::sqdiff_op{},
                                                                     stream,
                                                                     centroids.data_handle(),
                                                                     newCentroids.data_handle());

        DataT sqrdNormError = 0;
        raft::copy(&sqrdNormError, sqrdNorm.data_handle(), sqrdNorm.size(), stream);

        raft::copy(
            centroidsRawData.data_handle(), newCentroids.data_handle(), newCentroids.size(), stream);


        bool done = false;
        if (params.inertia_check) {
            // calculate cluster cost phi_x(C)
            detail::computeClusterCost(handle,
                                                                 minClusterAndDistance.view(),
                                                                 workspace,
                                                                 raft::make_device_scalar_view(clusterCostD.data()),
                                                                 raft::value_op{},
                                                                 raft::add_op{});

            DataT curClusteringCost = clusterCostD.value(stream);

            ASSERT(curClusteringCost != (DataT)0.0,
                         "Too few points and centroids being found is getting 0 cost from "
                         "centers");

            if (n_iter[0] > 1) {
                DataT delta = curClusteringCost / priorClusteringCost;
                //if (delta > 1 - params.tol) done = true;
            }
            priorClusteringCost = curClusteringCost;
        }

        resource::sync_stream(handle, stream);
        if ((check_converged) && sqrdNormError < params.tol) done = true;

        if (done) {
            RAFT_LOG_DEBUG("Threshold triggered after %d iterations. Terminating early.", n_iter[0]);
            break;
        }
#if LOG
        centroids_out<<"END ITERATION "<<(n_iter[0]-1)<<std::endl;
#endif
    }
#if LOG
    centroids_out.close();
#endif

    n_iter[0]--;

    std::cout<<std::fixed<<"centroids-update-time: "<<update_time<<"s"<<std::endl;
    std::cout<<std::fixed<<"dist-argmin-time: "<<dist_time<<"s"<<std::endl;
    std::cout<<"n_iter[0] "<<n_iter[0]<<std::endl;

    auto centroids = raft::make_device_matrix_view<DataT, IndexT>(
        centroidsRawData.data_handle(), n_clusters, n_features);

    detail::minClusterAndDistanceCompute<DataT, IndexT>(handle,
                                                                                                            X,
                                                                                                            centroids,
                                                                                                            minClusterAndDistance.view(),
                                                                                                            l2normx_view,
                                                                                                            L2NormBuf_OR_DistBuf,
                                                                                                            params.metric,
                                                                                                            params.batch_samples,
                                                                                                            params.batch_centroids,
                                                                                                            workspace);

    // TODO: add different templates for InType of binaryOp to avoid thrust transform
    thrust::transform(resource::get_thrust_policy(handle),
                                        minClusterAndDistance.data_handle(),
                                        minClusterAndDistance.data_handle() + minClusterAndDistance.size(),
                                        weight.data_handle(),
                                        minClusterAndDistance.data_handle(),
                                        [=] __device__(const raft::KeyValuePair<IndexT, DataT> kvp, DataT wt) {
                                            raft::KeyValuePair<IndexT, DataT> res;
                                            res.value = kvp.value * wt;
                                            res.key     = kvp.key;
                                            return res;
                                        });

    // calculate cluster cost phi_x(C)
    detail::computeClusterCost(handle,
                                                         minClusterAndDistance.view(),
                                                         workspace,
                                                         raft::make_device_scalar_view(clusterCostD.data()),
                                                         raft::value_op{},
                                                         raft::add_op{});

    inertia[0] = clusterCostD.value(stream);

    RAFT_LOG_DEBUG("KMeans.fit: completed after %d iterations with %f inertia[0] ",
                                 n_iter[0] > params.max_iter ? n_iter[0] - 1 : n_iter[0],
                                 inertia[0]);
}
}
}
}



template <typename T>
void read_svm(const uint32_t n, const uint32_t d,
                std::ifstream& in, T * d_dataset)
{
    T * h_dataset = new T[d*n];

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


    cudaMemcpy(d_dataset, h_dataset, sizeof(T)*d*n, cudaMemcpyHostToDevice);
	delete[] h_dataset;

}


template <typename T>
void read_csv(const uint32_t n, const uint32_t d,
				std::istream& in, T * d_dataset)
{

    T * h_dataset = new T[d*n];

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
    /*
    for (size_t i = 0; i <= n; i++) {
    char str[MAX_LINE] = { 0 };
    in >> str;

    if (i == (size_t)0) { continue; }
    if (!str[0]) { break; }

    int j = 0;
    char *tok = strtok(str, SEPARATOR);
    while (tok && j < d) {
      h_dataset[j + (i-1)*d] = atof(tok);
      tok = strtok(NULL, SEPARATOR);
      j++;
    }

    }
  */

    /*
    std::ofstream points_out;
    points_out.open("points-raft.out");
    for (int i=0; i<n; i++) {
      for (int j=0; j<d; j++) {
          points_out<<h_dataset[j + i*d]<<",";
      }
      points_out<<std::endl;
    }
    points_out.close();
    */

    cudaMemcpy(d_dataset, h_dataset, sizeof(T)*d*n, cudaMemcpyHostToDevice);
    delete[] h_dataset;
}



void run_kmeans(const uint32_t n, const uint32_t d, const uint32_t k, const uint32_t n_iters, const bool check_converged,
                float tol,
                std::string infile)
{
    using namespace raft;
    
    typedef float data_t; 
    typedef uint32_t ind_t;

    std::ifstream istream;
    istream.open(infile);

	std::random_device rd;
	std::mt19937 gen(rd());


    const raft::resources handle;
    cluster::KMeansParams params;
    params.n_clusters = k;
    params.max_iter = n_iters;
    params.init = cluster::KMeansParams::InitMethod::Random;
    params.tol = tol;
    params.inertia_check = true;

    auto centroids = raft::make_device_matrix<data_t, ind_t>(handle, k, d);
    auto points = raft::make_device_matrix<data_t, ind_t>(handle, n, d);


    if (infile.compare("-1")==0) { 
        raft::random::RngState rand(1234ULL);
        raft::random::uniform(handle, rand,
                        raft::make_device_vector_view(points.data_handle(), points.size()),
                        (data_t)-1e5, (data_t)1e5);
    } else if (infile.find("svm")!=std::string::npos) {
        read_svm(n, d, istream, points.data_handle());
    } else if (infile.find("csv")!=std::string::npos) {
        read_csv(n, d, istream, points.data_handle());
    }

    istream.close();


    auto points_view = raft::make_device_matrix_view<const data_t>(points.data_handle(),
                                                                    n, d);
    auto weight = raft::make_device_vector<data_t, ind_t>(handle, n);
    thrust::fill(raft::resource::get_thrust_policy(handle),
                    weight.data_handle(),
                    weight.data_handle() + weight.size(),
                    1);

    cudaStream_t stream = raft::resource::get_cuda_stream(handle);
    rmm::device_uvector<char> workspace(0, stream);

    data_t inertia; ind_t n_iter_run;

    std::cout<<"Running kmeans"<<std::endl;
    std::cout<<"n:"<<n<<" d:"<<d<<" k:"<<k<<std::endl;

    double score = 0;

    double kmeans_time = 0;
    for (int i=0; i<n_trials; i++) {
        params.rng_state.seed = gen();
        cluster::detail::initRandom<data_t, ind_t>(handle, params, points_view, centroids.view());

        auto stime = std::chrono::system_clock::now();
        cluster::detail::my_kmeans_fit_main<data_t, ind_t>
                            (handle,
                             params,
                             points_view,
                             weight.view(),
                             centroids.view(),
                             raft::make_host_scalar_view(&inertia),
                             raft::make_host_scalar_view(&n_iter_run),
                             workspace,
                             check_converged);
        auto etime = std::chrono::system_clock::now();
        auto kmeans_duration = std::chrono::duration_cast<std::chrono::duration<double>>(etime - stime);
        if (i>0) {
            kmeans_time += kmeans_duration.count();
        }
        score += inertia;
        std::cout<<inertia<<std::endl;
    }

    kmeans_time /= (n_trials - 1);
    score /= (n_trials);



    double fused_dist_time = 0;
    /*
    {
        auto minClusterAndDistance = raft::make_device_vector<raft::KeyValuePair<ind_t, data_t>, ind_t>(handle, n);

        rmm::device_uvector<data_t> buf(0, stream);

        // Run fused distances-argmin kernel
        for (int i=0; i<n_iter_run; i++) {
            auto stime = std::chrono::system_clock::now();

            auto l2Norm = raft::make_device_vector<data_t, ind_t>(handle, n);
            linalg::rowNorm(l2Norm.data_handle(), points.data_handle(), points.extent(1), points.extent(0),
                            linalg::L2Norm, true, stream); 
            auto l2Norm_view = raft::make_device_vector_view<const data_t>(l2Norm.data_handle(),
                                                                        n);
            cluster::detail::minClusterAndDistanceCompute<data_t, ind_t>
                    (
                    handle,
                    points_view,
                    centroids.view(),
                    minClusterAndDistance.view(),
                    l2Norm_view,
                    buf,
                    distance::DistanceType::L2Expanded,
                    n,
                    k,
                    workspace);
            resource::sync_stream(handle, stream);
            auto etime = std::chrono::system_clock::now();
            auto fused_duration = std::chrono::duration_cast<std::chrono::duration<double>>(etime - stime);
            fused_dist_time += fused_duration.count();
        }

    }*/

    // Run distances and argmin separately
    double pw_dist_time = 0;
    double argmin_time = 0;
    /*
    {

        auto pwDist = raft::make_device_matrix<data_t, ind_t>(handle, n, k);

        auto minClusterAndDistance = raft::make_device_vector<raft::KeyValuePair<ind_t, data_t>, ind_t>(handle, n);

        raft::KeyValuePair<ind_t, data_t> init(0, std::numeric_limits<data_t>::max());

        for (int i=0; i<n_iter_run; i++) {
            std::cout<<"Iteration "<<i<<std::endl;
            auto stime = std::chrono::system_clock::now();
            cluster::detail::pairwise_distance_kmeans<data_t, ind_t>
                    (
                    handle,
                    points_view,
                    centroids.view(),
                    pwDist.view(),
                    workspace,
                    distance::DistanceType::L2Expanded);
            resource::sync_stream(handle, stream);
            auto etime = std::chrono::system_clock::now();
            auto pw_duration = std::chrono::duration_cast<std::chrono::duration<double>>(etime - stime);
            pw_dist_time += pw_duration.count();

            stime = std::chrono::system_clock::now();
            linalg::coalescedReduction
                                        (minClusterAndDistance.data_handle(),
                                        pwDist.data_handle(),
                                        pwDist.extent(1), pwDist.extent(0),
                                        init,
                                        stream, true,
                                        [=] __device__(const data_t val, const ind_t i) {
                                            raft::KeyValuePair<ind_t, data_t> pair;
                                            pair.key   = i;
                                            pair.value = val;
                                            return pair;
                                        },
                                        raft::argmin_op{},
                                        raft::identity_op{});
            etime = std::chrono::system_clock::now();
            auto amin_duration = std::chrono::duration_cast<std::chrono::duration<double>>(etime - stime);
            argmin_time += amin_duration.count();
        }
    }

    */

    std::cout<<std::fixed<<"kmeans-time: "<<kmeans_time<<"s"<<std::endl;
    std::cout<<std::fixed<<"kmeans-score: "<<score<<std::endl;
    std::cout<<std::fixed<<"kmeans-iterations: "<<n_iter_run<<std::endl;
                                        
}



int main(int argc, char ** argv)
{
    int n = std::atoi(argv[1]);
    int d = std::atoi(argv[2]);
    int k = std::atoi(argv[3]);
    int n_iters = std::atoi(argv[4]);
    bool check_converged = (bool)std::atoi(argv[5]);
    std::string infile;
    if (argc > 6)
        infile = std::string(argv[6]);
    else
        infile = std::string("-1");
    run_kmeans(n, d, k, n_iters, check_converged,0.0001, infile);
    return 0;
}
