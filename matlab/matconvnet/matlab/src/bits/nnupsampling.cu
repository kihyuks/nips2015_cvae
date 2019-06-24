// @file nnupsampling.cu
// @brief Unpooling block
// @author Xinchen Yan
// Modified by Xinchen Yan, Feb. 19, 2015
//#include <iostream>
#include "nnupsampling.hpp"
#include "impl/upsampling.hpp"

#if ENABLE_GPU
#include "datacu.hpp"
#endif

//TODO: enable cudnn??

#include <assert.h>

using namespace vl ;

/* nnupsampling_forward */

Error
vl::nnupsampling_forward(vl::Context& context,
                        vl::Tensor output,
                        vl::Tensor data,
                        bool sparse,
                        int strideY, int strideX)
{
  Error status = vlSuccess ;
  switch (output.getMemoryType()) {
    default:
      assert(false);
      return vl::vlErrorUnknown ;
    case vl::CPU:
      status = vl::impl::upsampling_forward<CPU, float>
      ((float*)output.getMemory(), (float const*)data.getMemory(),
       sparse,
       data.getHeight(), data.getWidth(), data.getDepth() * data.getSize(),
       strideY, strideX) ;
      break;

#ifdef ENABLE_GPU
    case vl::GPU:
      //std::cerr << "nnupsampling.cu: " << sparse << std::endl;//TODO
      status = vl::impl::upsampling_forward<GPU, float>
      ((float*)output.getMemory(), (float const*)data.getMemory(),
      sparse,
      data.getHeight(), data.getWidth(), data.getDepth() * data.getSize(),
      strideY, strideX) ;
      
      if (status == vlErrorCuda) {
        context.setError(context.getCudaHelper().catchCudaError("upsampling_forward")) ;
      }
      break;
#endif
  }
  return context.passError(status, "nnupsampling_forward: ") ;
}

/* nnupsampling_backward */
Error
vl::nnupsampling_backward(Context& context,
                         Tensor derData,
                         Tensor derUpsampled,
                         bool sparse,
                         int strideY, int strideX)
{
  vl::Error status = vlSuccess ;
  switch (derData.getMemoryType()) {
    default:
      assert(faulse) ;
      return vl::vlErrorUnknown ;
    case vl::CPU:
      status = vl::impl::upsampling_backward<CPU, float>
      ((float*)derData.getMemory(), (float const*)derUpsampled.getMemory(),
       sparse,
       derData.getHeight(), derData.getWidth(), derData.getDepth() * derData.getSize(),
       strideY, strideX) ;
      break;

# if ENABLE_GPU
    case vl::GPU:
      status = vl::impl::upsampling_backward<GPU, float>
      ((float*)derData.getMemory(), (float const*)derUpsampled.getMemory(),
      sparse,
      derData.getHeight(), derData.getWidth(), derData.getDepth() * derData.getSize(),
      strideY, strideX);

      if (status == vlErrorCuda) {
        context.setError(context.getCudaHelper().catchCudaError("upsampling_backward: ")) ;
      }

      break;
#endif
  }
  return context.passError(status, "nnupsampling_backward: ") ;

}
