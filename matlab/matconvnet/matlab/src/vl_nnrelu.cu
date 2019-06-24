// @file vl_nnrelu.cu
// @brief Relu non-linearity implementation (fast computation)
// @author Xinchen Yan
// Created by Xinchen Yan, March. 23, 2015 (in compatible with matconvnet 1.0.9)

#include "bits/mexutils.h"
#include "bits/datamex.hpp"
#include "bits/nnrelu.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <assert.h>

/* option codes */
enum {
  opt_verbose,
  opt_cudnn,
  opt_no_cudnn,
};

/* options */
vlmxOption options [] = {
  {"Verbose",         0,  opt_verbose         },
  {"CUDNN",           0,  opt_cudnn           },
  {"NoCUDNN",         0,  opt_no_cudnn        },
  {0,                 0,  0                   }
} ;

/* ------------------------------------------------------------- */
/*                                                       Context */
/* ------------------------------------------------------------- */

vl::MexContext context;

/*
 Resetting the context here resolves a crash when MATLAB quits and
 the ~Context function is implicitly called on unloading the MEX file.
*/

void atExit()
{
  context.clear() ;
}

/* ------------------------------------------------------------- */
/*                                                    MEX driver */
/* ------------------------------------------------------------- */

enum {
  IN_DATA = 0,  IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  bool backMode = false ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  /* check the arguments */

  mexAtExit(atExit);

  if (nin < 1) {
    mexErrMsgTxt("The arguments are less than one.") ;
  }

  if (nin > 1 && vlmxIsString(in[1],-1)) {
    next = 1;
    backMode = 0;
  } else {
    backMode = (nin >= 2) ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose :
        ++ verbosity ;
        break;

      case opt_no_cudnn :
#if ENABLE_CUDNN
        context.getCudaHelper().setCudnnEnabled(false);
#endif
        break;

      case opt_cudnn:
#if ENABLE_CUDNN
        context.getCudaHelper().setCudnnEnabled(true);
#endif
        break;

      default: break;
    }
  }

  vl::MexTensor data(context);
  vl::MexTensor derOutput(context);

  data.init(in[IN_DATA]) ;
  if (backMode) {derOutput.init(in[IN_DEROUTPUT]) ; }

  if (backMode && ! vl::areCompatible(data, derOutput)) {
    mexErrMsgTxt("DATA and DEROUTPUT are not both CPU or GPU arrays.") ;
  }

  /* Get the output geometry */
  vl::TensorGeometry outputGeom(data.getHeight(), data.getWidth(), data.getDepth(), data.getSize());

  if (backMode && (derOutput != outputGeom)) {
    mexErrMsgTxt("DEROUTPUT dimensions are incompatible with X.");
  }

  /* Create output buffers */
  vl::Device type = data.getMemoryType();
  vl::MexTensor output(context);
  vl::MexTensor derData(context);

  if (!backMode) {
    output.init(type, outputGeom, 0);
  } else {
    derData.init(type, data.getGeometry(), 0);
  }

  if (verbosity > 0) {
    mexPrintf("vl_nnrelu: %s; %s", backMode?"backward":"forward", (data.getMemoryType()==vl::GPU) ? "GPU": "CPU") ;
    if (data.getMemoryType() == vl::GPU) {
#if ENABLE_CUDNN
      mexPrintf("; %s\n", context.getCudaHelper().getCudnnEnabled() ? "cuDNN" : "MatConvNet") ;
#else
      mexPrintf("; MatConvNet\n") ;
#endif
    } else {
      mexPrintf("; MatConvNet\n");
    }
    vl::print("vl_nnrelu: data: ", data) ;
    if (backMode) {
      vl::print("vl_nnrelu: derOutput: ", derOutput) ;
      vl::print("vl_nnrelu: derData", derData) ;
    } else {
      vl::print("vl_nnrelu: output: ", output) ;
    }
  }

  /*  Do the work */

  vl::Error error;
  if (!backMode) {
    error = vl::nnrelu_forward(context, output, data);
  } else {
    error = vl::nnrelu_backward(context, derData, data, derOutput);
  }

  /* Finish  */

  if (error != vl::vlSuccess) {
    mexErrMsgTxt(context.getLastErrorMessage().c_str());
  }
  if (backMode) {
    out[OUT_RESULT] = derData.relinquish();
  } else {
    out[OUT_RESULT] = output.relinquish();
  }
}
