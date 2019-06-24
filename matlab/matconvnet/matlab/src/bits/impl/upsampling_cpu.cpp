// @file upsampleing_cpu.cpp
// @brief Upsampling block implementation (CPU)
// @author Xinchen Yan
// Created by Xinchen Yan, Apr. 23, 2019 (in compatible with matconvnet 1.0.9)


#include "upsampling.hpp"
#include "../data.hpp"
#include <algorithm>
#include <limits>


/* ---------------------------------------------------------------- */
/*                                                 Upsampling (CPU) */
/* ---------------------------------------------------------------- */

template<typename type> static inline void
upsampling_forward_cpu(type* upsampled,
                       type const* data,
                       bool sparse,
                       size_t width, size_t height, size_t depth,
                       size_t strideX, size_t strideY)
{
  int upsampledWidth = width * strideX ;
  int upsampledHeight = height * strideY ;

  for (int z = 0; z < depth; ++z) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int x1 = x * (signed)strideX ;
        int y1 = y * (signed)strideY ;
        int x2 = (x + 1) * (signed)strideX;
        int y2 = (y + 1) * (signed)strideY;
        
        for (int v = y1 ; v < y2 ; ++v) {
          for (int u = x1 ; u < x2 ; ++u) {
            if (sparse)
              upsampled[v * upsampledWidth + u] = (type)0 ;
            else
              upsampled[v * upsampledWidth + u] = data[y * width + x];
          }
        }
        upsampled[y1 * upsampledWidth + x1] = data[y * width + x] ;
      }
    }
    data += width*height ;
    upsampled += upsampledWidth*upsampledHeight ;
  }   
}

template<> vl::Error
vl::impl::upsampling_forward<vl::CPU, float>(float* upsampled,
                                             float const* data,
                                             bool sparse,
                                             size_t height, size_t width, size_t depth,
                                             size_t strideY, size_t strideX) 
{
  upsampling_forward_cpu<float>(upsampled,
                                data,
                                sparse,
                                height, width, depth,
                                strideY, strideX);
  return vlSuccess;
}

/* ---------------------------------------------------------------- */
/*                                         UpsamplingBackward (CPU) */
/* ---------------------------------------------------------------- */

/* 
 assume the output array to be cleared or otherwise
 properly initialised: accumulates the derivative
 */
template<typename type> static inline void
upsampling_backward_cpu(type* derData,
                        type const* derUpsampled,
                        bool sparse,
                        size_t width, size_t height, size_t depth,
                        size_t strideX, size_t strideY)
{
  int upsampledWidth = width * strideX ;
  int upsampledHeight = height * strideY ;

  for (int z = 0; z < depth; ++z) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int x1 = x * (int)strideX ;
        int y1 = y * (int)strideY ;
        int x2 = (x + 1) * (int)strideX;
        int y2 = (y + 1) * (int)strideY;
        
        if (sparse) {
          derData[y * width + x] += derUpsampled[y1 * upsampledWidth + x1] ;
        } else {
          for (int v = y1 ; v < y2 ; ++v) {
            for (int u = x1 ; u < x2 ; ++u) {
              derData[y * width + x] += derUpsampled[v * upsampledWidth + u];
            }
          }
          //Average
          derData[y * width + x] /= (type)(strideX * strideY);
        }

      }
    }
    derData += width*height ;
    derUpsampled += upsampledWidth*upsampledHeight ;
  }
   
}

template<> vl::Error
vl::impl::upsampling_backward<vl::CPU, float>(float* derData,
                                              float const* derUpsampled,
                                              bool sparse,
                                              size_t height, size_t width, size_t depth,
                                              size_t strideY, size_t strideX) 

{
  upsampling_backward_cpu<float> (derData, derUpsampled,
                                  sparse,
                                  height, width, depth,
                                  strideY, strideX);
  return vlSuccess;
}


