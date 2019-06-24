// @file unpooling.hpp
// @brief Unpooling block implementation
// @author Xinchen yan
// Modified by Xinchen Yan, Feb. 19, 2015 (in compatible with matconvnet 1.0.9)

#ifndef VL_NNPOOLING_H
#define VL_NNPOOLING_H

#include "../data.hpp"
#include <cstddef>

namespace vl { namespace impl {

  template<vl::Device dev, typename type> vl::Error
  unpooling_forward(type* unpooled,
                    type const* data,
                    size_t height, size_t width, size_t depth,
                    size_t strideY, size_t strideX);
  
  template<vl::Device dev, typename type> vl::Error
  unpooling_backward(type* derData,
                     type const* derUnpooled,
                     size_t height, size_t width, size_t depth,
                     size_t strideY, size_t strideX);
  
  /* Specializations: CPU, float */

  template<> vl::Error
  unpooling_forward<vl::CPU, float>(float* unpooled,
                             float const* data,
                             size_t height, size_t width, size_t depth,
                             size_t strideY, size_t strideX);

  template<> vl::Error
  unpooling_backward<vl::CPU, float>(float* derData,
                              float const* derUnpooled,
                              size_t height, size_t width, size_t depth,
                              size_t strideY, size_t strideX);

  /* Specializations: GPU, float */

#if ENABLE_GPU
  template<> vl::Error
  unpooling_forward<vl::GPU, float>(float* unpooled,
                             float const* data,
                             size_t height, size_t width, size_t depth,
                             size_t strideY, size_t strideX);

  template<> vl::Error
  unpooling_backward<vl::GPU, float>(float* derData,
                              float const* derUnpooled,
                              size_t height, size_t width, size_t depth,
                              size_t strideY, size_t strideX);
#endif

} }

#endif /* defined(VL_NNPOOLING_H) */
