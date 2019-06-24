// @file vl_nnreshape.cu
// @brief Reshape implementation (fast computation)
// @author Xinchen Yan
// Created by Xinchen Yan, March. 23, 2015 (in compatible with matconvnet 1.0.9)
#include <iostream>
//TODO
#include "bits/mexutils.h"
#include "bits/datamex.hpp"
#include "bits/nnreshape.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif

#include <assert.h>

/* option codes */
enum {
  opt_verbose,
  opt_winsize,
  opt_cudnn,
  opt_no_cudnn,
};

/* options */
vlmxOption options [] = {
  {"Verbose",         0,  opt_verbose         },
  {"Winsize",         1,  opt_winsize         },
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
  IN_DATA = 0, IN_SIZE, IN_DEROUTPUT, IN_END
} ;

enum {
  OUT_RESULT = 0, OUT_END
} ;

void mexFunction(int nout, mxArray *out[],
                 int nin, mxArray const *in[])
{
  int resHeight;
  int resWidth;
  int resChannel;

  bool backMode = false ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  /* check the arguments */
  mexAtExit(atExit);
  
  if (nin < 2) {
    mexErrMsgTxt("The arguments are less than two.") ;
  }

  if (nin > 2 && vlmxIsString(in[2],-1)) {
    next = 2;
    backMode = 0;
  } else {
    backMode = (nin >= 3) ;
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

  if (!vlmxIsPlainMatrix(in[IN_SIZE],-1,-1)) {
     mexErrMsgTxt("WINSIZE is not a plain matrix.") ;
  }
  
  resHeight = mxGetPr(in[IN_SIZE])[0] ;
  resWidth = mxGetPr(in[IN_SIZE])[1] ;
  resChannel = mxGetPr(in[IN_SIZE])[2] ;

  if (resWidth * resHeight * resChannel != data.getHeight() * data.getWidth() * data.getDepth()) {
    mexErrMsgTxt("DATA dimensions are incompatible with OUTPUT.") ;
  }

  /* Get the output geometry */
  vl::TensorGeometry outputGeom(resHeight, resWidth, resChannel, data.getSize());

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
    mexPrintf("vl_nnreshape: %s; %s", backMode?"backward":"forward", (data.getMemoryType()==vl::GPU) ? "GPU": "CPU") ;
    if (data.getMemoryType() == vl::GPU) {
#if ENABLE_CUDNN
      mexPrintf("; %s\n", context.getCudaHelper().getCudnnEnabled() ? "cuDNN" : "MatConvNet") ;
#else
      mexPrintf("; MatConvNet\n") ;
#endif
    } else {
      mexPrintf("; MatConvNet\n");
    }
    vl::print("vl_nnreshape: data: ", data) ;
    if (backMode) {
      vl::print("vl_nnreshape: derOutput: ", derOutput) ;
    } else {
      vl::print("vl_nnreshape: output: ", output) ;
    }
  }

  /*  Do the work */
  vl::Error error;
  if (!backMode) {
    error = vl::nnreshape_forward(context, output, data);
  } else {
    error = vl::nnreshape_backward(context, derData, derOutput);
  }

  /* Finish */
  if (error != vl::vlSuccess) {
    mexErrMsgTxt(context.getLastErrorMessage().c_str());
  }
  if (backMode) {
    out[OUT_RESULT] = derData.relinquish();
  } else {
    out[OUT_RESULT] = output.relinquish();
  }
}
