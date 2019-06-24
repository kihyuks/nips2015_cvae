// @file upsampling.hpp
// @brief Upsampling filters
// @author Xinchen Yan
// Created by Xinchen Yan, Apr. 23, 2015 (in compatible with matconvnet 1.0.9)

#ifndef __vl__nnupsampling__
#define __vl__nnupsampling__

#include "data.hpp"
#include <stdio.h>

namespace vl {
  
  vl::Error
  nnupsampling_forward(vl::Context& context,
                      vl::Tensor output,
                      vl::Tensor data,
                      bool sparse,
                      int strideY, int strideX);
  vl::Error
  nnupsampling_backward(vl::Context& context,
                       vl::Tensor derData,
                       vl::Tensor derOutput,
                       bool sparse,
                       int strideY, int strideX);
}

#endif /* defined(__vl__nnupsampling__) */

