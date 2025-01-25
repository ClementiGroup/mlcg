#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <tuple>

// using namespace torch::indexing;
namespace mlcg_opt_radius {
    
std::tuple<torch::Tensor, torch::Tensor> 
              radius_cuda(const torch::Tensor x, 
                          torch::optional<torch::Tensor> ptr_x,
                          torch::optional<torch::Tensor> exclude_pair_xs,
                          torch::optional<torch::Tensor> ptr_exclude_pair_xs,
                          const double r,
                          const int64_t max_num_neighbors,
                          const bool ignore_same_index);    

std::tuple<torch::Tensor, torch::Tensor> 
exclusion_pair_to_ptr(const torch::Tensor exc_pair_index, 
                      const int64_t num_nodes) {
    TORCH_CHECK(exc_pair_index.dtype() == at::kLong);
    bool IsFromCPU = exc_pair_index.device().type() == at::DeviceType::CPU;
    const at::Tensor cpu_exc_pair_index = IsFromCPU ? exc_pair_index : exc_pair_index.to(at::kCPU);

    // TORCH_INTERNAL_ASSERT(exc_pair_index.device().type() == at::DeviceType::CPU);
    TORCH_CHECK(cpu_exc_pair_index.dim() == 2);
    TORCH_CHECK(cpu_exc_pair_index.size(0) == 2);
    at::Tensor a_contig = cpu_exc_pair_index.contiguous();
    // a lexsort over the pair indices
    const at::Tensor col_0 = a_contig.index({0});
    const at::Tensor col_1 = a_contig.index({1});
    at::Tensor indicex = at::argsort(col_0, true);
    const at::Tensor col_1_new = col_1.index({indicex});
    indicex = indicex.index({at::argsort(col_1_new, true)});
    at::Tensor pair_xs = col_0.index({indicex});
    const at::Tensor pair_ys = col_1.index({indicex});
    auto n_pairs = pair_xs.size(0);
    auto y_ptr = torch::empty(num_nodes + 1, cpu_exc_pair_index.options());
    const int64_t* pair_ys_arr = pair_ys.data_ptr<int64_t>();
    int64_t* y_ptr_arr = y_ptr.data_ptr<int64_t>();
    y_ptr_arr[0] = 0;
    int64_t i_pair = 0;
    int64_t current_pair_y;
    for (int64_t n_y = 0; n_y < num_nodes; n_y++) {
        while (i_pair < n_pairs) {
            current_pair_y = pair_ys_arr[i_pair];
            TORCH_CHECK(current_pair_y >= n_y);
            if (current_pair_y == n_y)
                i_pair++;
            else break;
        }
        y_ptr_arr[n_y + 1] = i_pair;
    }
    if (!IsFromCPU) {
        pair_xs = pair_xs.to(exc_pair_index);
        y_ptr = y_ptr.to(exc_pair_index);
    }
        
    return std::make_tuple(pair_xs, y_ptr);
}


PYBIND11_MODULE(mlcg_opt_radius, m){
    m.def("exclusion_pair_to_ptr", &exclusion_pair_to_ptr);
    m.def("radius_cuda", &radius_cuda);
}

static auto registry =
    torch::RegisterOperators().op("mlcg::exclusion_pair_to_ptr", &exclusion_pair_to_ptr);
}