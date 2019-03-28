
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <cuda.h>
#include <stdio.h>

#include "matirx_op.h"
#include "conv_op.h"
#include "reduce_op.h"

int main()
{
	test_vectorAdd();
	test_matrixMul();
	test_tiledMatrixMul();

	test_conv1d();
	test_tiledConstantConv2d();

	test_sumReduceShared();
	test_calHist();
    return 0;
}
