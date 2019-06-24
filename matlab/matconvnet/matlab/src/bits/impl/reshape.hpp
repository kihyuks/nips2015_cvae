// @file reshape.hpp
// @brief Reshape implementation (fast computation)
// @author Xinchen Yan
// Modified by Xinchen Yan, Apr. 23, 2015 (in compatible with matconvnet 1.0.9)

#ifndef VL_NNRESHAPE_H
#define VL_NNRESHAPE_H

#include "../data.hpp"
#include <cstddef>

namespace vl { namespace impl {
  
  template<vl::Device dev, typename type> vl::Error
  reshape_forward(type* output, type const* data,
               size_t ndim, size_t ndata);
  
  template<vl::Device dev, typename type> vl::Error
  reshape_backward(type* derData, type const* derOutput,
                size_t ndim, size_t ndata);

  /* Specializations: CPU, float */

  template<> vl::Error
  reshape_forward<vl::CPU, float>(float* output, float const* data,
                                  size_t ndim, size_t ndata);

  template<> vl::Error
  reshape_backward<vl::CPU, float>(float* derData, float const* derOutput,
                                   size_t ndim, size_t ndata);

  /* Specializations: GPU, float */

#if ENABLE_GPU

  template<> vl::Error
  reshape_forward<vl::GPU, float>(float* output, float const* data,
                                  size_t ndim, size_t ndata);

  template<> vl::Error
  reshape_backward<vl::GPU, float>(float* derData, float const* derOutput,
                                   size_t ndim, size_t ndata);

#endif 

} }

#endif /* defined(VL_NNRESHAPE_H) */
