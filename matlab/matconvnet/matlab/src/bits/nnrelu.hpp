// @file relu.hpp
// @brief Relu non-linearity (fast computation)
// @author Xinchen Yan
// Created by Xinchen Yan, Apr. 23, 2015 (in compatible with matconvnet 1.0.9)

#ifndef __vl__nnrelu__
#define __vl__nnrelu__

#include "data.hpp"
#include <stdio.h>

namespace vl {

  vl::Error
  nnrelu_forward(vl::Context& context,
                 vl::Tensor output,
                 vl::Tensor data);

  vl::Error
  nnrelu_backward(vl::Context& context,
                  vl::Tensor derData,
                  vl::Tensor data, 
                  vl::Tensor derOutput);
}

#endif /* defined(__vl__nnrelu) */
