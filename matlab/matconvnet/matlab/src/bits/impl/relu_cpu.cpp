// @file relu_cpu.cpp
// @brief Relu non-linearity implementation (CPU)
// @author Xinchen Yan
// Created by Xinchen Yan, Apr. 23, 2015 (in compatible with matconvnet 1.0.9)

#include "relu.hpp"
#include "../data.hpp"
#include <algorithm>
#include <limits>

/* ------------------------------------------------------------ */
/*                                                   Relu (CPU) */
/* -------------------------------------------------------------*/

template<typename type> static inline void
relu_forward_cpu(type* output, type const* data,
                 size_t height, size_t width, size_t depth) 
{
  type zero = (type)0;
  for (int z = 0; z < depth; ++z) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        output[y * width + x] = std::max(zero, data[y * width + x]);
        //TODO: inline???
      }
    }
  }
  data += width * height;
  output += width * height;
}

template<> vl::Error
vl::impl::relu_forward<vl::CPU, float>(float* output, float const* data,
                                      size_t height, size_t width, size_t depth) 
{
  relu_forward_cpu<float>(output, data, height, width, depth);
  return vlSuccess;
}

/* ------------------------------------------------------------ */
/*                                           ReluBackward (CPU) */
/* -------------------------------------------------------------*/

/*
 * assume the output array to be cleared or otherwise 
 * properly initalized: accumulates the derivative
 */
template<typename type> static inline void
relu_backward_cpu(type* derData, type const* data, type const* derOutput,
                  size_t height, size_t width, size_t depth)
{
  type zero = (type)0;
  for (int z = 0; z < depth; ++z) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        if (data[y * width + x] > zero)
          derData[y * width + x] = derOutput[y * width + x];
        else
          derData[y * width + x] = zero;
      }
    }
    data += width * height;
    derData += width * height;
    derOutput += width * height;
  }
}

template<> vl::Error
vl::impl::relu_backward<vl::CPU, float>(float* derData, float const* data, float const* derOutput,
                                       size_t height, size_t width, size_t depth)
{
  relu_backward_cpu<float> (derData, data, derOutput, height, width, depth);
  return vlSuccess;
}

