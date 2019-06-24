// @file reshape_cpu.cpp
// @brief Reshape implementation (CPU)
// @author Xinchen Yan
// Created by Xinchen Yan, Apr. 23, 2015 (in compatible with matconvnet 1.0.9)

#include "reshape.hpp"
#include "../data.hpp"
#include <algorithm>
#include <limits>

/* ------------------------------------------------------------ */
/*                                                Reshape (CPU) */
/* -------------------------------------------------------------*/

template<typename type> static inline void
reshape_forward_cpu(type* output, type const* data,
                    size_t ndim, size_t ndata) 
{
  for (int idx_data = 0; idx_data < ndata; ++idx_data) {
    for (int idx_dim = 0; idx_dim < ndim; ++idx_dim) {
      output[idx_dim] = data[idx_dim];
    }
    data += ndim;
    output += ndim;
  }
}

template<> vl::Error
vl::impl::reshape_forward<vl::CPU, float>(float* output, float const* data,
                                          size_t ndim, size_t ndata) 
{
  reshape_forward_cpu<float>(output, data, 
                             ndim, ndata);
  return vlSuccess;
}

/* ------------------------------------------------------------ */
/*                                        ReshapeBackward (CPU) */
/* -------------------------------------------------------------*/

/*
 * assume the output array to be cleared or otherwise 
 * properly initalized: accumulates the derivative
 */
template<typename type> static inline void
reshape_backward_cpu(type* derData, type const* derOutput,
                     size_t ndim, size_t ndata)
{
 for (int idx_data = 0; idx_data < ndata; ++idx_data) {
    for (int idx_dim = 0; idx_dim < ndim; ++idx_dim) {
      derData[idx_dim] = derOutput[idx_dim];
    }
    derData += ndim;
    derOutput += ndim;
  }

}

template<> vl::Error
vl::impl::reshape_backward<vl::CPU, float>(float* derData, float const* derOutput,
                                           size_t ndim, size_t ndata)
{
  reshape_backward_cpu<float> (derData, derOutput,
                               ndim, ndata);
  return vlSuccess;
}

