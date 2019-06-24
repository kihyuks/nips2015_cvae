// @file nnunpooling.cu
// @brief Unpooling block
// @author Xinchen Yan
// Modified by Xinchen Yan, Feb. 19, 2015

#include "nnunpooling.hpp"
#include "impl/unpooling.hpp"

#if ENABLE_GPU
#include "datacu.hpp"
#endif

//TODO: enable cudnn??

#include <assert.h>

using namespace vl ;

/* nnpooling_forward */

Error
vl::nnunpooling_forward(vl::Context& context,
                        vl::Tensor output,
                        vl::Tensor data,
                        int strideY, int strideX)
{
  Error status = vlSuccess ;
  switch (output.getMemoryType()) {
    default:
      assert(false);
      return vl::vlErrorUnknown ;
    case vl::CPU:
      status = vl::impl::unpooling_forward<CPU, float>
      ((float*)output.getMemory(), (float const*)data.getMemory(),
       data.getHeight(), data.getWidth(), data.getDepth() * data.getSize(),
       strideY, strideX) ;
      break;

#ifdef ENABLE_GPU
    case vl::GPU:
      status = vl::impl::unpooling_forward<GPU, float>
      ((float*)output.getMemory(), (float const*)data.getMemory(),
      data.getHeight(), data.getWidth(), data.getDepth() * data.getSize(),
      strideY, strideX) ;
      
      if (status == vlErrorCuda) {
        context.setError(context.getCudaHelper().catchCudaError("unpooling_forward")) ;
      }
      break;
#endif
  }
  return context.passError(status, "nnunpooling_forward: ") ;
}

/* nnunpooling_backward */
Error
vl::nnunpooling_backward(Context& context,
                         Tensor derData,
                         Tensor derUnpooled,
                         int strideY, int strideX)
{
  vl::Error status = vlSuccess ;
  switch (derData.getMemoryType()) {
    default:
      assert(faulse) ;
      return vl::vlErrorUnknown ;
    case vl::CPU:
      status = vl::impl::unpooling_backward<CPU, float>
      ((float*)derData.getMemory(), (float const*)derUnpooled.getMemory(),
       derData.getHeight(), derData.getWidth(), derData.getDepth() * derData.getSize(),
       strideY, strideX) ;
      break;

# if ENABLE_GPU
    case vl::GPU:
      status = vl::impl::unpooling_backward<GPU, float>
      ((float*)derData.getMemory(), (float const*)derUnpooled.getMemory(),
      derData.getHeight(), derData.getWidth(), derData.getDepth() * derData.getSize(),
      strideY, strideX);

      if (status == vlErrorCuda) {
        context.setError(context.getCudaHelper().catchCudaError("unpooling_backward: ")) ;
      }

      break;
#endif
  }
  return context.passError(status, "nnunpooling_backward: ") ;

}
