// @file vl_nnunpool.cu
// @brief Unpooling block
// @author Xinchen Yan
// Modified by Xinchen Yan, Feb. 19, 2015 (in compatible with matconvnet 1.0.9)

#include "bits/mexutils.h"
#include "bits/datamex.hpp"
#include "bits/nnunpooling.hpp"

#if ENABLE_GPU
#include "bits/datacu.hpp"
#endif 

#include <assert.h>

/* option codes */
enum {
  opt_stride = 0,
  opt_verbose,
  opt_cudnn,
  opt_no_cudnn,
} ;

/* options */
vlmxOption  options [] = {
  {"Stride",           1,   opt_stride            },
  {"Verbose",          0,   opt_verbose           },
  {"CUDNN",            0,   opt_cudnn             },
  {"NoCUDNN",          0,   opt_no_cudnn          },
  {0,                  0,   0                     }
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
  int strideX = 1 ;
  int strideY = 1 ;

  bool backMode = false ;

  int verbosity = 0 ;
  int opt ;
  int next = IN_END ;
  mxArray const *optarg ;

  /* -------------------------------------------------------------- */
  /*                                            Check the arguments */
  /* -------------------------------------------------------------- */

  mexAtExit(atExit);

  if (nin < 1) {
    mexErrMsgTxt("The arguments are less than one.") ;
  }

  if (nin > 1 && vlmxIsString(in[1],-1)) {
    next = 1 ;
    backMode = 0 ;
  } else {
    backMode = (nin >= 2) ;
  }

  while ((opt = vlmxNextOption (in, nin, options, &next, &optarg)) >= 0) {
    switch (opt) {
      case opt_verbose :
        ++ verbosity ;
        break ;

      case opt_stride :
        if (!vlmxIsPlainMatrix(optarg,-1,-1)) {
          mexErrMsgTxt("STRIDE is not a plain matrix.") ;
        }
        switch (mxGetNumberOfElements(optarg)) {
          case 1:
            strideY = (int)mxGetPr(optarg)[0] ;
            strideX = strideY ;
            break ;
          case 2:
            strideY = (int)mxGetPr(optarg)[0] ;
            strideX = (int)mxGetPr(optarg)[1] ;
            break ;
          default:
            mexErrMsgTxt("STRIDE has neither one nor two elements.") ;
        }
        break ;
      case opt_no_cudnn:
#if ENABLE_CUDNN
        context.getCudaHelper().setCudnnEnabled(false);
#endif
        break;
      case opt_cudnn:
#if ENABLE_CUDNN
        context.getCudaHelper().setCudnnEnabled(true);
#endif
        break;

      default: break ;
    }
  }

  vl::MexTensor data(context);
  vl::MexTensor derOutput(context);

  data.init(in[IN_DATA]);
  if (backMode) { derOutput.init(in[IN_DEROUTPUT]); }
  if (backMode && ! vl::areCompatible(data, derOutput)) {
    mexErrMsgTxt("DATA and DEROUTPUT are not both CPU or GPU arrays.");
  }

  //if (!vlmxIsPlainMatrix(in[IN_SIZE],-1,-1)) {
  //  mexErrMsgTxt("SIZE is not a plain matrix.");
  //}

  /* Basic compatibility of geometry */
  if (strideX < 1 || strideY < 1) {
    mexErrMsgTxt("At least one element of STRIDE is smaller than one.");
  }

  /* Get the output geometry */
  vl::TensorGeometry outputGeom(data.getHeight() * strideY,
                                data.getWidth() * strideX,
                                data.getDepth(),
                                data.getSize());


  if (backMode && (derOutput != outputGeom)) {
    mexErrMsgTxt("DEROUTPUT dimensions are incompatible with X and POOL.");
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
    mexPrintf("vl_nnunpool: %s; %s", backMode?"backward":"forward", (data.getMemoryType()==vl::GPU)?"GPU":"CPU") ;
    if (data.getMemoryType() == vl::GPU) {
#if ENABLE_CUDNN
      mexPrintf("; %s\n", context.getCudaHelper().getCudnnEnabled()?"cuDNN":"MatConvNet");
#else
      mexPrintf("; MatconvNet\n");
#endif
    } else {
      mexPrintf("; MatconvNet\n");
    }
    mexPrintf("vl_nnunpool: stride: [%d %d]\n",
              strideY, strideX) ;

    vl::print("vl_nnunpool: data: ", data);
    
    if (backMode) {
      vl::print("vl_nnunpool: derOutput: ", derOutput);
      vl::print("vl_nnunpool: derData: ", derData);
    } else {
      vl::print("vl_nnunpool: output: ", output);
    }
  }

  /* -------------------------------------------------------------- */
  /*                                                    Do the work */
  /* -------------------------------------------------------------- */

  vl::Error error;
  if (!backMode) {
    error = vl::nnunpooling_forward(context,
                                    output, data,
                                    strideY, strideX);
  } else {
    error = vl::nnunpooling_backward(context,
                                     derData, derOutput,
                                     strideY, strideX);
  }
  /* -------------------------------------------------------------- */
  /*                                                         Finish */
  /* -------------------------------------------------------------- */

  if (error != vl::vlSuccess) {
    mexErrMsgTxt(context.getLastErrorMessage().c_str());
  }
  if (backMode) {
    out[OUT_RESULT] = derData.relinquish();
  } else {
    out[OUT_RESULT] = output.relinquish();
  }
}
