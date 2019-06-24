// @file nnreshape.cu
// @brief Reshape implementation (fast computation)
// @author Xinchen Yan
// Created by Xinchen Yan, Apr. 23, 2015

#include "nnreshape.hpp"
#include "impl/reshape.hpp"

#if ENABLE_GPU
#include "datacu.hpp"
#endif

//TODO: enable cudnn??

#include <assert.h>

using namespace vl ;

/* nnreshape_forward */

Error
vl::nnreshape_forward(vl::Context& context,
                   vl::Tensor output,
                   vl::Tensor data)
{
  Error status = vlSuccess;
  switch (output.getMemoryType()) {
    default:
      assert(false);
      return vl::vlErrorUnknown;
    case vl::CPU:
      status = vl::impl::reshape_forward<CPU, float>
      ((float*)output.getMemory(), (float const*)data.getMemory(),
       data.getHeight() * data.getWidth() * data.getDepth(), data.getSize());
      break;

#ifdef ENABLE_GPU
    case vl::GPU:
      status = vl::impl::reshape_forward<GPU, float>
      ((float*)output.getMemory(), (float const*)data.getMemory(),
       data.getHeight() * data.getWidth() * data.getDepth(), data.getSize());
      
      if (status == vlErrorCuda) {
        context.setError(context.getCudaHelper().catchCudaError("reshape_forward")) ;
      }
      break;
      
#endif 
  }
  return context.passError(status, "nnreshape_forward: ") ;
}

/* nnreshape_backward */
Error
vl::nnreshape_backward(vl::Context& context,
                    vl::Tensor derData,
                    vl::Tensor derOutput)
{
  vl::Error status = vlSuccess ;
  switch (derData.getMemoryType()) {
    default:
      assert(false) ;
      return vl::vlErrorUnknown;
    case vl::CPU:
      status = vl::impl::reshape_backward<CPU, float>
      ((float*)derData.getMemory(), (float const*)derOutput.getMemory(), 
       derData.getHeight() * derData.getWidth() * derData.getDepth(), derData.getSize());
      break;

#if ENABLE_GPU
    case vl::GPU:
      status = vl::impl::reshape_backward<GPU, float>
      ((float*)derData.getMemory(), (float const*)derOutput.getMemory(),
       derData.getHeight() * derData.getWidth() * derData.getDepth(), derData.getSize());

      if (status == vlErrorCuda) {
        context.setError(context.getCudaHelper().catchCudaError("reshape_backward: ")) ;
      }
      break;
#endif
  }
  return context.passError(status, "nnreshape_backward: ") ;
}

