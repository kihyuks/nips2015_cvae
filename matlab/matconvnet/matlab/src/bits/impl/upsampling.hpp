// @file upsampling.hpp
// @brief Upsampling block implementation
// @author Xinchen yan
// Created by Xinchen Yan, Mar. 23, 2015 (in compatible with matconvnet 1.0.9)

#ifndef VL_NNUPSAMPLING_H
#define VL_NNUPSAMPLING_H

#include "../data.hpp"
#include <cstddef>

namespace vl { namespace impl {

  template<vl::Device dev, typename type> vl::Error
  upsampling_forward(type* upsampled,
                     type const* data,
                     bool sparse,
                     size_t height, size_t width, size_t depth,
                     size_t strideY, size_t strideX);
 
  template<vl::Device dev, typename type> vl::Error
  upsampling_backward(type* derData,
                      type const* derUpsampled,
                      bool sparse,
                      size_t height, size_t width, size_t depth,
                      size_t strideY, size_t strideX);


  /* Specializations: CPU, float */

  template<> vl::Error
  upsampling_forward<vl::CPU, float>(float* upsampled,
                              float const* data,
                              bool sparse,
                              size_t height, size_t width, size_t depth,
                              size_t strideY, size_t strideX);

  template<> vl::Error
  upsampling_backward<vl::CPU, float>(float* derData,
                               float const* derUpsampled,
                               bool sparse,
                               size_t height, size_t width, size_t depth,
                               size_t strideY, size_t strideX);

  /* Specializations: GPU, float */

#if ENABLE_GPU
  template<> vl::Error
  upsampling_forward<vl::GPU, float>(float* upsampled,
                              float const* data,
                              bool sparse,
                              size_t height, size_t width, size_t depth,
                              size_t strideY, size_t strideX);

  template<> vl::Error
  upsampling_backward<vl::GPU, float>(float* derData,
                               float const* derUpsampled,
                               bool sparse,
                               size_t height, size_t width, size_t depth,
                               size_t strideY, size_t strideX);
#endif

} }

#endif /* defined(VL_NNUPSAMPLING_H) */
