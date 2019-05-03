/***********************************************************
tissueGPU4.cu
GPU kernel to perform matrix multiplcation in matrix tissue method.
Each tissue point is assigned one or more threads: step is the number of threads
This spreads it over more threads.
TWS February 2012
************************************************************/
#include <stdio.h>
#include <cutil_inline.h>


__global__ void tissueGPU4Kernel(int *d_tisspoints, float *d_dtt000, float *d_qtp000, float *d_xt, float *d_rt,
	int nnt, int step, float diff)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
	int itp = i/step;
    int itp1 = i%step;
	int jtp,ixyz,ix,iy,iz,nnt2=2*nnt,istep;
	float r = 0.;
    if(itp < nnt){
		ix = d_tisspoints[itp];
		iy = d_tisspoints[itp+nnt];
		iz = d_tisspoints[itp+nnt2];
		for(jtp=itp1; jtp<nnt; jtp+=step){
			ixyz = abs(d_tisspoints[jtp]-ix) + abs(d_tisspoints[jtp+nnt]-iy) + abs(d_tisspoints[jtp+nnt2]-iz);
			r -= d_dtt000[ixyz]*d_qtp000[jtp]*d_xt[jtp];
		}
		r /= diff;
		r += d_xt[itp];	//diagonal of matrix has 1s
		if(itp1 == 0) d_rt[itp] = r;
	}
	//The following is apparently needed to assure that d_pt000 is incremented in sequence from the needed threads
	for(istep=1; istep<step; istep++){
		__syncthreads();
		if(itp1 == istep && itp < nnt) d_rt[itp] += r;
	}
}

extern "C" void tissueGPU4(int *d_tisspoints, float *d_dtt000, float *d_qtp000, float *d_xt, float *d_rt,
	int nnt, int useGPU, float diff)
{
	int threadsPerBlock = 256;
	int step = 4;//has to be a power of two apparently
	int blocksPerGrid = (step*nnt + threadsPerBlock - 1) / threadsPerBlock;
	tissueGPU4Kernel<<<blocksPerGrid, threadsPerBlock>>>(d_tisspoints,d_dtt000,d_qtp000,d_xt,d_rt,nnt,step,diff);
}
