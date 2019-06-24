// @file reshape.hpp
// @brief Reshape (fast computation)
// @author Xinchen Yan
// Created by Xinchen Yan, Apr. 23, 2015 (in compatible with matconvnet 1.0.9)

#ifndef __vl__nnreshape__
#define __vl__nnreshape__

#include "data.hpp"
#include <stdio.h>

namespace vl {

  vl::Error
  nnreshape_forward(vl::Context& context,
                 vl::Tensor output,
                 vl::Tensor data);

  vl::Error
  nnreshape_backward(vl::Context& context,
                  vl::Tensor derData,
                  vl::Tensor derOutput);
}

#endif /* defined(__vl__nnreshape) */
