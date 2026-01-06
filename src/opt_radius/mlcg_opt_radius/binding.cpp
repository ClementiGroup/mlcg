#include <torch/extension.h>
#include <tuple>

namespace mlcg_opt_radius {
    
    std::tuple<torch::Tensor, torch::Tensor> 
    radius_cuda(const torch::Tensor x, 
                torch::optional<torch::Tensor> ptr_x,
                const double r,
                const int64_t max_num_neighbors,
                const bool ignore_same_index,
                torch::optional<torch::Tensor> exclude_pair_xs,
                torch::optional<torch::Tensor> ptr_exclude_pair_xs);

    std::tuple<torch::Tensor, torch::Tensor> 
    exclusion_pair_to_ptr(const torch::Tensor exc_pair_index, 
                        const int64_t num_nodes);

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
        m.def("exclusion_pair_to_ptr", &exclusion_pair_to_ptr);
        m.def("radius_cuda", &radius_cuda);
    }
}
