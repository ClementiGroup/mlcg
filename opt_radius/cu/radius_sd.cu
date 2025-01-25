#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h> 

#define THREADS 256 

#define CHECK_CUDA(x) AT_ASSERTM(x.device().is_cuda(), #x " must be CUDA tensor")
#define CHECK_INPUT(x) AT_ASSERTM(x, "Input mismatch")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")

namespace cg = cooperative_groups;

namespace mlcg_opt_radius {
//-------------------------------------------------------------------
__forceinline__ 
__device__ 
int64_t 
get_example_idx(int64_t idx,
                const int64_t *ptr,
                const int64_t num_examples) {
    for (int64_t i = 0; i < num_examples; i++) {
        if (ptr[i + 1] > idx)
            return i;
    }
    return num_examples - 1;
}
//-------------------------------------------------------------------
__forceinline__
__device__
int64_t
is_in_sorted_array(
                const int64_t query,
		const int64_t *arr_to_be_searched,
		int64_t& current_search_id,
                const int64_t max_search_id) {
    int64_t current_v;
    while (current_search_id < max_search_id) {
        current_v = arr_to_be_searched[current_search_id];
	if (current_v < query) {
            current_search_id++;
	}
	else if (current_v == query) {
            return true;
        }
	else return false; // pointing at a value bigger than the query
    }
    if (current_search_id == max_search_id) {
        return false; // we have drained the search
    }
}
//-------------------------------------------------------------------
template <typename scalar_t>
__global__ void
compact_kernel(const int64_t n,
               const int64_t max_num_neighbors,
               int64_t *__restrict__ t_save_idx,
               int64_t *__restrict__ o_edge_index,
               int64_t *__restrict__ o_c_edge_index,
               scalar_t *__restrict__ o_distance,
               scalar_t *__restrict__ o_c_distance) {

    const int64_t gid = blockIdx.x * blockDim.x + threadIdx.x;
    if (gid >= n)
        return;

    int64_t offset = 0;
    int64_t count = t_save_idx[0];
    if (gid > 0) {
        offset = t_save_idx[gid-1];
        count = (t_save_idx[gid] - t_save_idx[gid-1]);
    }

    for (int64_t i = 0; i < count; i++) {
        o_c_edge_index[(offset*2) + (2*i)    ] = o_edge_index[(gid * max_num_neighbors * 2) + (2*i)    ];
        o_c_edge_index[(offset*2) + (2*i) + 1] = o_edge_index[(gid * max_num_neighbors * 2) + (2*i) + 1];
        o_c_distance[offset + i] = o_distance[(gid * max_num_neighbors) + i];
    }
}

template <typename scalar_t>
__global__ void
radius_kernel(const scalar_t *__restrict__ x, 
              const int64_t *__restrict__ ptr_x,
              const int64_t *__restrict__ exclude_pair_xs,
	      const int64_t *__restrict__ ptr_exclude_pair_xs,
              const scalar_t r, 
              const int64_t n,
              const int64_t dim, 
              const int64_t num_examples,
              const int64_t max_num_neighbors,
              const bool ignore_same_index,
              int64_t *__restrict__ t_save_idx,
              int64_t *__restrict__ o_edge_index,
              scalar_t *__restrict__ o_distance) {

    cg::grid_group grid = cg::this_grid();

    const int64_t n_y = blockIdx.x * blockDim.x + threadIdx.x;
    if (n_y >= n)
        return;

    int64_t count = 0;
    const int64_t example_idx = get_example_idx(n_y, ptr_x, num_examples);

    int64_t exc_pair_i = ptr_exclude_pair_xs[n_y];
    int64_t bound_exc_pair_i = ptr_exclude_pair_xs[n_y + 1];

    for (int64_t n_x = ptr_x[example_idx]; n_x < ptr_x[example_idx + 1]; n_x++) {
        scalar_t dist = 0;
        for (int64_t d = 0; d < dim; d++) {
            dist += (x[n_x * dim + d] - x[n_y * dim + d]) *
                    (x[n_x * dim + d] - x[n_y * dim + d]);
        }
	bool is_excluded = is_in_sorted_array(
            n_x, exclude_pair_xs, exc_pair_i, bound_exc_pair_i

        );
        if (!is_excluded && (dist < r) && !(ignore_same_index && (n_y == n_x))) {
            o_edge_index[(n_y * max_num_neighbors * 2) + (count * 2)    ] = n_y;
            o_edge_index[(n_y * max_num_neighbors * 2) + (count * 2) + 1] = n_x;
            o_distance[(n_y * max_num_neighbors) + count] = sqrt(dist); //squared
            count++;
        }
        if (count >= max_num_neighbors)
            break;
    }

    // using prefix sum to compute save ids
    const int gid = n_y; // just for readability
    const int tid = threadIdx.x; 
    const int bid = blockIdx.x; 
    const int tbs = blockDim.x;
    extern __shared__ int64_t tb_counts[];
    tb_counts[tid] = count;
    __syncthreads();

    // within threadblock
    for (int offset = 1; offset < tbs; offset *= 2) {
        int64_t temp = 0;
        if (tid >= offset) {
            temp = tb_counts[tid - offset];
        }
        __syncthreads(); 
        tb_counts[tid] += temp;
        __syncthreads();
    }

    // send max per threadblock count to global level
    if (tid==0 && gid < n){
        t_save_idx[bid] = tb_counts[tbs-1];
    }
    grid.sync();
    //------------------------------------------------------------
    // sum the last element of each block
    if (bid == 0 && tid == 0) {
        for (int i = 1; i < gridDim.x; ++i) {
            t_save_idx[i] += t_save_idx[i - 1];
        }
    }
    grid.sync();
    // back to theadblock
    if (bid > 0 && gid < n)
        tb_counts[tid] += t_save_idx[bid-1];
    grid.sync();
    // save idx
    t_save_idx[gid] = tb_counts[tid];
    
    // from previous attempt, ignore:w
    //------------------------------------------------------------
    //if (tid == 0 && bid > 0) {
    //    int64_t offset = 0;
    //    for (int i = 0; i < bid; ++i) {
    //        offset += t_save_idx[i];
    //    }
    //    if (gid < n)
    //        tb_counts[tid] += offset;
    //}
    //------------------------------------------------------------
}

std::tuple<torch::Tensor, torch::Tensor> 
              radius_cuda(const torch::Tensor x, 
                          torch::optional<torch::Tensor> ptr_x,
                          torch::optional<torch::Tensor> exclude_pair_xs,
                          torch::optional<torch::Tensor> ptr_exclude_pair_xs,
                          const double r,
                          const int64_t max_num_neighbors,
                          const bool ignore_same_index) {
    CHECK_CUDA(x);
    CHECK_CONTIGUOUS(x);
    CHECK_INPUT(x.dim() == 2);
    c10::cuda::MaybeSetDevice(x.get_device());

    if (ptr_x.has_value()) {
        CHECK_CUDA(ptr_x.value());
        CHECK_INPUT(ptr_x.value().dim() == 1);
    } else
        ptr_x = torch::arange(0, x.size(0) + 1, x.size(0),
                              x.options().dtype(torch::kLong));
    
    if (exclude_pair_xs.has_value()) {
        CHECK_CUDA(exclude_pair_xs.value());
        CHECK_INPUT(exclude_pair_xs.value().dim() == 1);
        CHECK_INPUT(ptr_exclude_pair_xs.has_value());
        CHECK_CUDA(ptr_exclude_pair_xs.value());
        CHECK_INPUT(ptr_exclude_pair_xs.value().dim() == 1);
        CHECK_INPUT(ptr_exclude_pair_xs.value().size(0) == x.size(0) + 1);
    } else {
        // no exclusion? Generate an empty input
        exclude_pair_xs = torch::empty(1, x.options().dtype(torch::kLong));
        ptr_exclude_pair_xs = torch::zeros(x.size(0) + 1, x.options().dtype(torch::kLong));
    }
        

    auto scalar_type = x.scalar_type();
    dim3 BLOCKS((x.size(0) + THREADS - 1) / THREADS);
    auto stream = at::cuda::getCurrentCUDAStream();
    auto shared_memory_size = THREADS*sizeof(int64_t);

    auto o_edge_index = 
        torch::empty({x.size(0) * max_num_neighbors, 2}, ptr_x.value().options());
    auto o_distance = 
        torch::empty(x.size(0) * max_num_neighbors, x.options());
    auto t_save_idx = 
        torch::empty(x.size(0), ptr_x.value().options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, scalar_type, "_", [&] {

            scalar_t* x_data_ptr = x.data_ptr<scalar_t>();
            int64_t* ptr_x_ptr = ptr_x.value().data_ptr<int64_t>();
            int64_t* exclude_pair_xs_ptr = exclude_pair_xs.value().data_ptr<int64_t>();
            int64_t* ptr_exclude_pair_xs_ptr = ptr_exclude_pair_xs.value().data_ptr<int64_t>();
            scalar_t r_squared = r*r;
            int64_t x_size0 = x.size(0);
            int64_t x_size1 = x.size(1);
            int64_t max_ptr_x = ptr_x.value().numel() - 1;
            int64_t* t_save_idx_ptr = t_save_idx.data_ptr<int64_t>();
            int64_t* o_edge_index_ptr = o_edge_index.data_ptr<int64_t>();
            scalar_t* o_distance_ptr = o_distance.data_ptr<scalar_t>();
            void* kernel_args[] = {
                (void*)&x_data_ptr,
                (void*)&ptr_x_ptr,
                (void*)&exclude_pair_xs_ptr,
                (void*)&ptr_exclude_pair_xs_ptr,
                (void*)&r_squared,  
                (void*)&x_size0,    
                (void*)&x_size1,    
                (void*)&max_ptr_x,
                (void*)&max_num_neighbors,
                (void*)&ignore_same_index,
                (void*)&t_save_idx_ptr,
                (void*)&o_edge_index_ptr,
                (void*)&o_distance_ptr
            };
            void* kernel_func = (void*) &radius_kernel<scalar_t>;
            cudaError_t status = cudaLaunchCooperativeKernel(
                kernel_func,  // Kernel function pointer
                BLOCKS,  // Grid size (blocks)
                THREADS, // Block size (threads)
                kernel_args,   // Kernel arguments as an array of pointers
                shared_memory_size,  // Shared memory size
                stream  // CUDA stream
            );
            if (status != cudaSuccess) {
                throw std::runtime_error("Cooperative kernel launch failed: " + 
                    std::string(cudaGetErrorString(status)));
            }
        }
    );

    int64_t last_idx = t_save_idx.index({-1}).item<int64_t>();
    auto o_c_edge_index = 
        torch::empty({last_idx, 2}, ptr_x.value().options());
    auto o_c_distance = 
        torch::empty(last_idx, x.options());

    AT_DISPATCH_FLOATING_TYPES_AND2(
        at::ScalarType::Half, at::ScalarType::BFloat16, scalar_type, "_", [&] {
        compact_kernel<<<BLOCKS, THREADS, 0, stream>>>(
            x.size(0),
            max_num_neighbors,
            t_save_idx.data_ptr<int64_t>(),
            o_edge_index.data_ptr<int64_t>(),
            o_c_edge_index.data_ptr<int64_t>(),
            o_distance.data_ptr<scalar_t>(),
            o_c_distance.data_ptr<scalar_t>()
        );
    });
    return std::make_tuple(o_c_edge_index.t(), o_c_distance);
}

//PYBIND11_MODULE(mlcg_opt_radius, m){
//    m.def("radius_cuda", &radius_cuda);
//}

static auto registry =
    torch::RegisterOperators().op("mlcg::radius_cuda", &radius_cuda);

}