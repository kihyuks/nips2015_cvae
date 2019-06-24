// @file unpooling.hpp
// @brief Unpooling filters
// @author Xinchen Yan
// Modified by Xinchen Yan, Feb 19, 2015 (in compatible with matconvnet 1.0.9)

#ifndef __vl__nnunpooling__
#define __vl__nnunpooling__

#include "data.hpp"
#include <stdio.h>

namespace vl {
  
  vl::Error
  nnunpooling_forward(vl::Context& context,
                      vl::Tensor output,
                      vl::Tensor data,
                      int strideY, int strideX);
  vl::Error
  nnunpooling_backward(vl::Context& context,
                       vl::Tensor derData,
                       vl::Tensor derOutput,
                       int strideY, int strideX);
}

#endif /* defined(__vl__nnunpooling__) */

