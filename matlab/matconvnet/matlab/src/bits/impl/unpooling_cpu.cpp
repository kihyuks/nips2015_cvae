// @file unpooling_cpu.cpp
// @brief Unpooling block implementation (CPU)
// @author Xinchen Yan
// Modified by Xinchen Yan, Feb. 19, 2019 (in compatible with matconvnet 1.0.9)


#include "unpooling.hpp"
#include "../data.hpp"
#include <algorithm>
#include <limits>


/* ---------------------------------------------------------------- */
/*                                                 Unpooling (CPU) */
/* ---------------------------------------------------------------- */

template<typename type> static inline void
unpooling_forward_cpu(type* unpooled,
                            type const* data,
                            size_t width, size_t height, size_t depth,
                            size_t strideX, size_t strideY)
{
  int unpooledWidth = width * strideX ;
  int unpooledHeight = height * strideY ;

  for (int z = 0; z < depth; ++z) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int x1 = x * (signed)strideX ;
        int y1 = y * (signed)strideY ;
        int x2 = (x + 1) * (signed)strideX;
        int y2 = (y + 1) * (signed)strideY ;
            
        for (int v = y1 ; v < y2 ; ++v) {
          for (int u = x1 ; u < x2 ; ++u) {
            unpooled[v * unpooledWidth + u] = (type)0 ;
          }
        }
        unpooled[y1 * unpooledWidth + x1] = data[y * width + x] ;
      }
    }
    data += width*height ;
    unpooled += unpooledWidth*unpooledHeight ;
  }   
}

template<> vl::Error
vl::impl::unpooling_forward<vl::CPU, float>(float* unpooled,
                                         float const* data,     
                                         size_t height, size_t width, size_t depth,
                                         size_t strideY, size_t strideX) 
{
  unpooling_forward_cpu<float>(unpooled,
                               data,
                               height, width, depth,
                               strideY, strideX);
  return vlSuccess;
}

/* ---------------------------------------------------------------- */
/*                                         UnpoolingBackward (CPU) */
/* ---------------------------------------------------------------- */

/* 
 assume the output array to be cleared or otherwise
 properly initialised: accumulates the derivative
 */
template<typename type> static inline void
unpooling_backward_cpu(type* derData,
                       type const* derUnpooled,
                       size_t width, size_t height, size_t depth,
                       size_t strideX, size_t strideY)
{
  int unpooledWidth = width * strideX ;
  int unpooledHeight = height * strideY ;

  for (int z = 0; z < depth; ++z) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        int x1 = x * (int)strideX ;
        int y1 = y * (int)strideY ;
        
        derData[y * width + x] += derUnpooled[y1 * unpooledWidth + x1] ;
      }
    }
    derData += width*height ;
    derUnpooled += unpooledWidth*unpooledHeight ;
  }
   
}

template<> vl::Error
vl::impl::unpooling_backward<vl::CPU, float>(float* derData,
                                      float const* derUnpooled,
                                      size_t height, size_t width, size_t depth,
                                      size_t strideY, size_t strideX) 

{
  unpooling_backward_cpu<float> (derData, derUnpooled,
                                 height, width, depth,
                                 strideY, strideX);
  return vlSuccess;
}


