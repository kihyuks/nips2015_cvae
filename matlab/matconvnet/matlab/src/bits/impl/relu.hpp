// @file relu.hpp
// @brief Relu non-linearity implementation (fast computation)
// @author Xinchen Yan
// Modified by Xinchen Yan, Apr. 23, 2015 (in compatible with matconvnet 1.0.9)

#ifndef VL_NNRELU_H
#define VL_NNRELU_H

#include "../data.hpp"
#include <cstddef>

namespace vl { namespace impl {
  
  template<vl::Device dev, typename type> vl::Error
  relu_forward(type* output, type const* data,
               size_t height, size_t width, size_t depth);
  
  template<vl::Device dev, typename type> vl::Error
  relu_backward(type* derData, type const* data, type const* derOutput,
                size_t height, size_t width, size_t depth);

  /* Specializations: CPU, float */

  template<> vl::Error
  relu_forward<vl::CPU, float>(float* output, float const* data,
                               size_t height, size_t width, size_t depth);

  template<> vl::Error
  relu_backward<vl::CPU, float>(float* derData, float const* data, float const* derOutput,
                                size_t height, size_t width, size_t depth);

  /* Specializations: GPU, float */

#if ENABLE_GPU

  template<> vl::Error
  relu_forward<vl::GPU, float>(float* output, float const* data,
                               size_t height, size_t width, size_t depth);

  template<> vl::Error
  relu_backward<vl::GPU, float>(float* derData, float const* data, float const* derOutput,
                                size_t height, size_t width, size_t depth);

#endif 

} }

#endif /* defined(VL_NNRELU_H) */
