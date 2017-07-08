////////////////////////////////////////////////////////////////////////////
//
// Copyright 1993-2015 NVIDIA Corporation.  All rights reserved.
//
// Please refer to the NVIDIA end user license agreement (EULA) associated
// with this source code for terms and conditions that govern your use of
// this software. Any use, reproduction, disclosure, or distribution of
// this software and related documentation outside the terms of the EULA
// is strictly prohibited.
//
////////////////////////////////////////////////////////////////////////////

/* Template project which demonstrates the basics on how to setup a project
* example application.
* Host code.
*/

// includes, system
//#include <stdlib.h>
#include <stdio.h>
//#include <string.h>
//#include <math.h>
#include <sys/time.h>

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <ctime>
#include <cfloat>

#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// includes CUDA
#include <cuda_runtime.h>

// includes, project
//#include <helper_cuda.h>
//#include <helper_functions.h> // helper functions for SDK examples

#include <cublas_v2.h>
#include <cudnn.h>

#define index5(i1,i2,i3,i4,i5,l1,l2,l3,l4) (i1*l1+i2*l2+i3*l3+i4*l4+i5)
#define index8(i1,i2,i3,i4,i5,i6,i7,i8,l1,l2,l3,l4,l5,l6,l7) (i1*l1+i2*l2+i3*l3+i4*l4+i5*l5+i6*l6+i7*l7+i8)

//#define layout1
//#define DEBUG

double timing(){
        double time;
        struct timeval timmer;

        gettimeofday(&timmer,NULL);
        time = timmer.tv_sec + timmer.tv_usec*1e-6;
        return time;
}


#define FatalError(s) do {                                             \
    std::stringstream _where, _message;                                \
    _where << __FILE__ << ':' << __LINE__;                             \
    _message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
    std::cerr << _message.str() << "\nAborting...\n";                  \
    cudaDeviceReset();                                                 \
    exit(1);                                                           \
} while(0)

#define checkCUDNN(status) do {                                        \
    std::stringstream _error;                                          \
    if (status != CUDNN_STATUS_SUCCESS) {                              \
      _error << "CUDNN failure: " << cudnnGetErrorString(status);      \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

#define checkCudaErrors(status) do {                                   \
    std::stringstream _error;                                          \
    if (status != 0) {                                                 \
      _error << "Cuda failure: " << status;                            \
      FatalError(_error.str());                                        \
    }                                                                  \
} while(0)

////////////////////////////////////////////////////////////////////////////////
// declaration, forward
void runTest(int argc, char **argv);

extern "C"
void computeGold(float *reference, float *idata, const unsigned int len);

struct ConvBiasLayer
{
    int in_channels, out_channels, kernel_size;
    int in_width, in_height, out_width, out_height;

    std::vector<float> pconv, pbias;

    ConvBiasLayer(int in_channels_, int out_channels_, int kernel_size_,
                  int in_w_, int in_h_) : pconv(in_channels_ * kernel_size_ * kernel_size_ * out_channels_),
                  pbias(out_channels_)
    {
        in_channels = in_channels_;
        out_channels = out_channels_;
        kernel_size = kernel_size_;
        in_width = in_w_;
        in_height = in_h_;
        out_width = in_w_ - kernel_size_ + 1;
        out_height = in_h_ - kernel_size_ + 1;
    }
};


inline bool
sdkCompareL2fe(const float *reference, const float *data,
               const unsigned int len, const float epsilon)
{
    //assert(epsilon >= 0);

    float error = 0;
    float ref = 0;

    for (unsigned int i = 0; i < len; ++i)
    {

        float diff = reference[i] - data[i];
        error += diff * diff;
        ref += reference[i] * reference[i];
    }

    float normRef = sqrtf(ref);

    if (fabs(ref) < 1e-7)
    {
#ifdef _DEBUG
        std::cerr << "ERROR, reference l2-norm is 0\n";
#endif
        return false;
    }

    float normError = sqrtf(error);
    error = normError / normRef;
    bool result = error < epsilon;
#ifdef _DEBUG

    if (! result)
    {
        std::cerr << "ERROR, l2-norm error "
                  << error << " is greater than epsilon " << epsilon << "\n";
    }

#endif

    return result;
}




inline void image_slice(int d, int D, int outTile, int imTile, int pad_d, int *start_d, int *stop_d, int *tile_pad_d1, int *tile_pad_d2)
{
	start_d[0] = d*outTile - pad_d;
	stop_d[0] = start_d[0] + imTile-1;
	tile_pad_d1[0] = 0;
	tile_pad_d2[0] = 0;
	if(start_d[0]<0){
		tile_pad_d1[0] = -start_d[0];
		start_d[0] = 0;
	}
	if(stop_d[0]>D-1){
		tile_pad_d2[0] = stop_d[0]-D+1;
		stop_d[0] = D-1;
	}
}

inline __device__ void image_slice_d(int d, int D, int outTile, int imTile, int pad_d, int *start_d, int *stop_d, int *tile_pad_d1, int *tile_pad_d2)
{
	start_d[0] = d*outTile - pad_d;
	stop_d[0] = start_d[0] + imTile-1;
	tile_pad_d1[0] = 0;
	tile_pad_d2[0] = 0;
	if(start_d[0]<0){
		tile_pad_d1[0] = -start_d[0];
		start_d[0] = 0;
	}
	if(stop_d[0]>D-1){
		tile_pad_d2[0] = stop_d[0]-D+1;
		stop_d[0] = D-1;
	}
}


__global__ void
FilterTransform_4x4x4_3x3x3(float *K_d, int K, int C, int T, int R, int S, int ktileSize, float *F_trans_d)
{
		int tid  = threadIdx.x;
	    int blkK = gridDim.x - blockIdx.x - 1;
	    int c    = gridDim.y - blockIdx.y - 1;
	    int k    = (blkK<<5) + tid;

	    bool valid_k = k < K;

#ifdef layout1 //KCTRS order to store filter data
	    int RS = R*S;
	    int TRS = T*RS;
	    int CTRS = C*T*R*S;
	    int offset = k*CTRS+c*T*R*S;

	    int TRSt = ktileSize*ktileSize*ktileSize;
	    //Out += blkK*C*32*TRSt + c*32*TRSt + tid;

	    int offsetT = -RS;
	    int offsetR = -S;

	    float TT[3][3][3];

	    for(int t=0;t<T;t++)
	    	for(int r=0;r<R;r++)
	    		for(int s=0;s<S;s++){
	    			float val = 0.0;
	    			if(valid_k)
	    				val = K_d[index5(k,c,t,r,s,CTRS,T*R*S,R*S,S)];
	    			TT[t][r][s] = val;
	    		}
#else  //CTRSK order to store filter
	    int SK = S*K;
	    int RSK = R*SK;
	    int TRSK = T*RSK;
	    float TT[3][3][3];
	    for(int t=0;t<T;t++)
	    	for(int r=0;r<R;r++)
	    		for(int s=0;s<S;s++){
	    			float val = 0.0;
	    			if(valid_k)
	    				val = K_d[index5(c,t,r,s,k,TRSK,RSK,SK,K)];
	    			TT[t][r][s] = val;
	    		}

#endif

	    float f25_88 = 25.0/88.0;
	    float f25_132 = 25.0/132.0;
	    float f25_198 = 25.0/198.0;
	    float f125_308 = 125.0/308;
	    float f400_441 = 400.0/441.0;
	    float f625_1078 = 625.0/1078.0;

	    float T0[3][6][3];
	    float T1[3][6][6];

	    for(int i=0;i<3;i++){
			#pragma unroll
	    	for(int j=0;j<3;j++){
	    		float t0 = f25_88*TT[i][2][j];
	    		float t1 = __fmaf_rn(TT[i][0][j],-f625_1078,-t0);
	    		float t2 = __fmaf_rn(TT[i][0][j],  f25_198,    t0);
	    		T0[i][0][j] = f400_441 * TT[i][0][j];
	    		T0[i][1][j] = __fmaf_rn(TT[i][1][j], -f125_308, t1);
	    		T0[i][2][j] = __fmaf_rn(TT[i][1][j],  f125_308, t1);
	    		T0[i][3][j] = __fmaf_rn(TT[i][1][j],  f25_132,  t2);
	    		T0[i][4][j] = __fmaf_rn(TT[i][1][j], -f25_132,  t2);
	    		T0[i][5][j] = TT[i][2][j];
	    	}
	    }
	    for(int i=0;i<3;i++){
			#pragma unroll
	    	for(int j=0;j<6;j++){
	    		float t0 = f25_88 *  T0[i][j][2];
	    		float t1 = __fmaf_rn(T0[i][j][0], -f625_1078, -t0);
	    		float t2 = __fmaf_rn(T0[i][j][0],  f25_198,    t0);
	    		T1[i][j][0] = f400_441 * T0[i][j][0];
	    		T1[i][j][1] = __fmaf_rn(T0[i][j][1], -f125_308, t1);
	    		T1[i][j][2] = __fmaf_rn(T0[i][j][1],  f125_308, t1);
	    		T1[i][j][3] = __fmaf_rn(T0[i][j][1],  f25_132,  t2);
	    		T1[i][j][4] = __fmaf_rn(T0[i][j][1], -f25_132,  t2);
	    		T1[i][j][5] = T0[i][j][2];
	    	}
	    }

	    int l4 = K;
	    int l3 = C*K;
	    int l2 = ktileSize*l3;
	    int l1 = ktileSize*l2;
	    for(int i=0;i<6;i++){
	    	#pragma unroll
	    	for(int j=0;j<6;j++) {
	    		float t0 = f25_88 *  T1[0][i][j];
	    		float t1 = __fmaf_rn(T1[0][i][j], -f625_1078, -t0);
	    		float t2 = __fmaf_rn(T1[0][i][j],  f25_198,    t0);
	    		//if(valid_k){
	    		F_trans_d[index5(0,i,j,c,k,l1,l2,l3,l4)] = f400_441 * T1[0][i][j];
	    		F_trans_d[index5(1,i,j,c,k,l1,l2,l3,l4)] = __fmaf_rn(T1[1][i][j], -f125_308, t1);
	    		F_trans_d[index5(2,i,j,c,k,l1,l2,l3,l4)] = __fmaf_rn(T1[1][i][j],  f125_308, t1);
	    		F_trans_d[index5(3,i,j,c,k,l1,l2,l3,l4)] = __fmaf_rn(T1[1][i][j],  f25_132,  t2);
	    		F_trans_d[index5(4,i,j,c,k,l1,l2,l3,l4)] = __fmaf_rn(T1[1][i][j], -f25_132,  t2);
	    		F_trans_d[index5(5,i,j,c,k,l1,l2,l3,l4)] = T1[2][i][j];//}
	    	}
	    }
}



__global__ void ImageTransform_4x4x4_3x3x3_d(float *I_h, float *I_trans_h,int imTile,int outTile,int dTileNum,int hTileNum,int wTileNum,int pad_d,int pad_h,int pad_w, int N, int C, int D, int H, int W)
{

    int start[3], stop[3], tile_pad1[3], tile_pad2[3];
	int dhwTileNum = dTileNum*hTileNum*wTileNum;

	float I[6][6][6];
	bool dv[6],hv[6],wv[6];


	int tid  = threadIdx.x;
	int blkN = gridDim.x - blockIdx.x - 1;
	int blkXYZ = gridDim.y - blockIdx.y - 1;
	int c    = gridDim.z - blockIdx.z - 1;
	int n    = (blkN<<5) + tid;

	bool valid_n = n < N;

	int blockDHW = blkXYZ;
	int d = (blockDHW/(wTileNum*hTileNum));
	int h = (blockDHW%(wTileNum*hTileNum))/wTileNum;
	int w = blockDHW%wTileNum;
	image_slice_d(d, D, outTile, imTile, pad_d, &start[0], &stop[0], &tile_pad1[0], &tile_pad2[0]);
	image_slice_d(h, H, outTile, imTile, pad_h, &start[1], &stop[1], &tile_pad1[1], &tile_pad2[1]);
	image_slice_d(w, W, outTile, imTile, pad_w, &start[2], &stop[2], &tile_pad1[2], &tile_pad2[2]);
	int start_d=start[0], stop_d=stop[0], tile_pad_d1=tile_pad1[0], tile_pad_d2=tile_pad2[0];
	int start_h=start[1], stop_h=stop[1], tile_pad_h1=tile_pad1[1], tile_pad_h2=tile_pad2[1];
	int start_w=start[2], stop_w=stop[2], tile_pad_w1=tile_pad1[2], tile_pad_w2=tile_pad2[2];

#ifdef layout1
	int HW = H*W;
	int DHW = D*HW;
	int CDHW = C*DHW;
#else
	int WN = W*N;
	int HWN = H*WN;
	int DHWN = D*HWN;
#endif
	for(int i1=0;i1<imTile;i1++){
		dv[i1]=0;
		if((i1-tile_pad_d1)<0 || (i1+tile_pad_d2)>=imTile)
			dv[i1]=1;
		for(int i2=0;i2<imTile;i2++) {
			hv[i2] = 0;
			if((i2-tile_pad_h1)<0 || (i2+tile_pad_h2)>=imTile)
				hv[i2] = 1;
			for(int i3=0;i3<imTile;i3++) {
				wv[i3] = 0;
				if((i3-tile_pad_w1)<0 || (i3+tile_pad_w2>=imTile))
					wv[i3] = 1;
				if((dv[i1]==1) || (hv[i2]==1) || (wv[i3]==1))
					I[i1][i2][i3] = 0;
				else{
#ifdef layout1  //NCDHW
					//I[i1][i2][i3] = I_h[n*C*D*H*W+c*D*H*W+(start_d+i1-tile_pad_d1)*H*W+(start_h+i2-tile_pad_h1)*W+start_w+i3-tile_pad_w1];//I_h[][][start_d+t1-tile_pad_d1][start_h+t2-tile_pad_h1][start_w+t3-tile_pad_w1]
					I[i1][i2][i3] = I_h[index5(n,c,(start_d+i1-tile_pad_d1),(start_h+i2-tile_pad_h1),(start_w+i3-tile_pad_w1),CDHW,DHW,HW,W)];
#else		//CDHWN
				    I[i1][i2][i3] = I_h[index5(c,(start_d+i1-tile_pad_d1),(start_h+i2-tile_pad_h1),(start_w+i3-tile_pad_w1),n,DHWN,HWN,WN,N)];//I_h[][][start_d+t1-tile_pad_d1][start_h+t2-tile_pad_h1][start_w+t3-tile_pad_w1]
#endif
				}
			}
		}
	}
	//image transforming
	float f1_1025 = 1.1025;
	float f2_74   = 2.7400;
	float f0_70   = 0.7000;
	float f0_49   = 0.4900;
	float T0[6][6][6];
	float T1[6][6][6];

	for(int i=0;i<imTile;i++){
		#pragma unroll
		for(int j=0;j<imTile;j++){
			float t0 = __fmaf_rn(I[2][i][j], -2.25f, I[4][i][j]);
			float t1 = __fmaf_rn(I[1][i][j], -2.25f, I[3][i][j]);
			float t2 = __fmaf_rn(I[2][i][j], -f0_49, I[4][i][j]);
			float t3 = __fmaf_rn(I[1][i][j], -f0_49, I[3][i][j]);
			float t4 = __fmaf_rn(I[2][i][j], -f2_74, I[4][i][j]);
			float t5 = __fmaf_rn(I[3][i][j], -f2_74, I[5][i][j]);

			T0[0][i][j] = __fmaf_rn(I[0][i][j], f1_1025, t4);
			T0[1][i][j] = __fmaf_rn(t1,  f0_70, t0);
			T0[2][i][j] = __fmaf_rn(t1, -f0_70, t0);
			T0[3][i][j] = __fmaf_rn(t3,  1.5f,  t2);
			T0[4][i][j] = __fmaf_rn(t3, -1.5f,  t2);
			T0[5][i][j] = __fmaf_rn(I[1][i][j], f1_1025, t5);
		}
	}
	for(int i=0;i<imTile;i++){
		#pragma unroll
		for(int j=0;j<imTile;j++){
			float t0 = __fmaf_rn(T0[i][2][j], -2.25f, T0[i][4][j]);
			float t1 = __fmaf_rn(T0[i][1][j], -2.25f, T0[i][3][j]);
			float t2 = __fmaf_rn(T0[i][2][j], -f0_49, T0[i][4][j]);
			float t3 = __fmaf_rn(T0[i][1][j], -f0_49, T0[i][3][j]);
			float t4 = __fmaf_rn(T0[i][2][j], -f2_74, T0[i][4][j]);
			float t5 = __fmaf_rn(T0[i][3][j], -f2_74, T0[i][5][j]);

			T1[i][0][j] = __fmaf_rn(T0[i][0][j], f1_1025, t4);
			T1[i][1][j] = __fmaf_rn(t1,  f0_70, t0);
			T1[i][2][j] = __fmaf_rn(t1, -f0_70, t0);
			T1[i][3][j] = __fmaf_rn(t3,  1.5f,  t2);
			T1[i][4][j] = __fmaf_rn(t3, -1.5f,  t2);
			T1[i][5][j] = __fmaf_rn(T0[i][1][j], f1_1025, t5);
		}
	}
#ifdef layout1

	int l7 = wTileNum;
	int l6 = hTileNum*l7;
	int l5 = dTileNum*l6;
	int l4 = C*l5;
	int l3 = N*l4;
	int l2 = imTile*l3;
	int l1 = imTile*l2;


	for(int i=0;i<imTile;i++){
		#pragma unroll
		for(int j=0;j<imTile;j++){
			float t0 = __fmaf_rn(T1[i][j][2], -2.25f, T1[i][j][4]);
			float t1 = __fmaf_rn(T1[i][j][1], -2.25f, T1[i][j][3]);
			float t2 = __fmaf_rn(T1[i][j][2], -f0_49, T1[i][j][4]);
			float t3 = __fmaf_rn(T1[i][j][1], -f0_49, T1[i][j][3]);
			float t4 = __fmaf_rn(T1[i][j][2], -f2_74, T1[i][j][4]);
			float t5 = __fmaf_rn(T1[i][j][3], -f2_74, T1[i][j][5]);

			//if(valid_n){
			I_trans_h[index8(i,j,0,n,c,d,h,w,l1,l2,l3,l4,l5,l6,l7)] = __fmaf_rn(T1[i][j][0], f1_1025, t4);
			I_trans_h[index8(i,j,1,n,c,d,h,w,l1,l2,l3,l4,l5,l6,l7)] = __fmaf_rn(t1,  f0_70, t0);
			I_trans_h[index8(i,j,2,n,c,d,h,w,l1,l2,l3,l4,l5,l6,l7)] = __fmaf_rn(t1, -f0_70, t0);
			I_trans_h[index8(i,j,3,n,c,d,h,w,l1,l2,l3,l4,l5,l6,l7)] = __fmaf_rn(t3,  1.5f,  t2);
			I_trans_h[index8(i,j,4,n,c,d,h,w,l1,l2,l3,l4,l5,l6,l7)] = __fmaf_rn(t3, -1.5f,  t2);
			I_trans_h[index8(i,j,5,n,c,d,h,w,l1,l2,l3,l4,l5,l6,l7)] = __fmaf_rn(T1[i][j][1], f1_1025, t5);//}
		}
	}
#else

	int l7 = N;
	int l6 = wTileNum*l7;
	int l5 = hTileNum*l6;
	int l4 = dTileNum*l5;
	int l3 = C*l4;
	int l2 = imTile*l3;
	int l1 = imTile*l2;

	for(int i=0;i<4;i++){
		#pragma unroll
		for(int j=0;j<4;j++){
			float t0 = __fmaf_rn(T1[i][j][2], -2.25f, T1[i][j][4]);
			float t1 = __fmaf_rn(T1[i][j][1], -2.25f, T1[i][j][3]);
			float t2 = __fmaf_rn(T1[i][j][2], -f0_49, T1[i][j][4]);
			float t3 = __fmaf_rn(T1[i][j][1], -f0_49, T1[i][j][3]);
			float t4 = __fmaf_rn(T1[i][j][2], -f2_74, T1[i][j][4]);
			float t5 = __fmaf_rn(T1[i][j][3], -f2_74, T1[i][j][5]);

			//if(valid_n){
			I_trans_h[index8(i,j,0,c,d,h,w,n,l1,l2,l3,l4,l5,l6,l7)] = __fmaf_rn(T1[i][j][0], f1_1025, t4);
			I_trans_h[index8(i,j,1,c,d,h,w,n,l1,l2,l3,l4,l5,l6,l7)] = __fmaf_rn(t1,  f0_70, t0);
			I_trans_h[index8(i,j,2,c,d,h,w,n,l1,l2,l3,l4,l5,l6,l7)] = __fmaf_rn(t1, -f0_70, t0);
			I_trans_h[index8(i,j,3,c,d,h,w,n,l1,l2,l3,l4,l5,l6,l7)] = __fmaf_rn(t3,  1.5f,  t2);
			I_trans_h[index8(i,j,4,c,d,h,w,n,l1,l2,l3,l4,l5,l6,l7)] = __fmaf_rn(t3, -1.5f,  t2);
			I_trans_h[index8(i,j,5,c,d,h,w,n,l1,l2,l3,l4,l5,l6,l7)] = __fmaf_rn(T1[i][j][1], f1_1025, t5);//}
		}
	}
#endif
}







__global__ void OutputTransform_4x4x4_3x3x3_d(float *O_winograd_h, float *O_trans_h, int imTile, int outTile, int dTileNum, int hTileNum,int wTileNum, int N, int K, int M, int P, int Q)
{

	float T0[6][4][6];
	float T1[6][4][4];
	float T2[4][4][4];


	int tid  = threadIdx.x;
	int blkK = gridDim.x - blockIdx.x - 1;
	int blkXYZ = gridDim.y - blockIdx.y - 1;
	int n    = gridDim.z - blockIdx.z - 1;
	int k    = (blkK<<5) + tid;

	bool valid_k = k < K;

	int blockDHW = blkXYZ;
	int d = (blockDHW/(wTileNum*hTileNum));
	int h = (blockDHW%(wTileNum*hTileNum))/wTileNum;
	int w = blockDHW%wTileNum;

	int dhwTileNum = dTileNum*hTileNum*wTileNum;
	int hwTileNum = hTileNum*wTileNum;
	int imTile3 = imTile*imTile*imTile;
	int imTile2 = imTile*imTile;
	int PQ = P*Q;
	int MPQ = M*PQ;
	int KMPQ = K*MPQ;

#ifdef layout1

	int l7 = K;
	int l6 = wTileNum*l7;
	int l5 = hTileNum*l6;
	int l4 = dTileNum*l5;
	int l3 = N*l4;
	int l2 = imTile*l3;
	int l1 = imTile*l2;

	for(int i=0;i<imTile;i++){
		#pragma unroll
		for(int j=0;j<imTile;j++){
			float t0 = O_trans_h[index8(i,1,j,n,d,h,w,k,l1,l2,l3,l4,l5,l6,l7)]+\
					   O_trans_h[index8(i,2,j,n,d,h,w,k,l1,l2,l3,l4,l5,l6,l7)];//I[i][1][j] + I[i][2][j];
			float t1 = O_trans_h[index8(i,3,j,n,d,h,w,k,l1,l2,l3,l4,l5,l6,l7)]+\
					   O_trans_h[index8(i,4,j,n,d,h,w,k,l1,l2,l3,l4,l5,l6,l7)];//I[i][3][j] + I[i][4][j];
			float t2 = O_trans_h[index8(i,1,j,n,d,h,w,k,l1,l2,l3,l4,l5,l6,l7)]-\
					   O_trans_h[index8(i,2,j,n,d,h,w,k,l1,l2,l3,l4,l5,l6,l7)];//I[i][1][j] - I[i][2][j];
			float t3 = O_trans_h[index8(i,3,j,n,d,h,w,k,l1,l2,l3,l4,l5,l6,l7)]-\
					   O_trans_h[index8(i,4,j,n,d,h,w,k,l1,l2,l3,l4,l5,l6,l7)];//I[i][3][j] - I[i][4][j];
			T0[i][0][j] = t0+t1+O_trans_h[index8(i,0,j,n,d,h,w,k,l1,l2,l3,l4,l5,l6,l7)];//I[i][0][j];
			T0[i][1][j] = t2*0.700 + t3*1.500;
			T0[i][2][j] = t0*0.490 + t1*2.250;
			T0[i][3][j] = t2*0.343 + t3*3.375 + O_trans_h[index8(i,5,j,n,d,h,w,k,l1,l2,l3,l4,l5,l6,l7)];//I[i][5][j];
		}
	}
#else
	int l7 = K;
	int l6 = N*l7;
	int l5 = wTileNum*l6;
	int l4 = hTileNum*l5;
	int l3 = dTileNum*l4;
	int l2 = imTile*l3;
	int l1 = imTile*l2;


	for(int i=0;i<imTile;i++){
			#pragma unroll
			for(int j=0;j<imTile;j++){
				//if(valid_k){
				float t0 = O_trans_h[index8(i,1,j,d,h,w,n,k,l1,l2,l3,l4,l5,l6,l7)]+\
						   O_trans_h[index8(i,2,j,d,h,w,n,k,l1,l2,l3,l4,l5,l6,l7)];//I[i][1][j] + I[i][2][j];
				float t1 = O_trans_h[index8(i,3,j,d,h,w,n,k,l1,l2,l3,l4,l5,l6,l7)]+\
						   O_trans_h[index8(i,4,j,d,h,w,n,k,l1,l2,l3,l4,l5,l6,l7)];//I[i][3][j] + I[i][4][j];
				float t2 = O_trans_h[index8(i,1,j,d,h,w,n,k,l1,l2,l3,l4,l5,l6,l7)]-\
						   O_trans_h[index8(i,2,j,d,h,w,n,k,l1,l2,l3,l4,l5,l6,l7)];//I[i][1][j] - I[i][2][j];
				float t3 = O_trans_h[index8(i,3,j,d,h,w,n,k,l1,l2,l3,l4,l5,l6,l7)]-\
						   O_trans_h[index8(i,4,j,d,h,w,n,k,l1,l2,l3,l4,l5,l6,l7)];//I[i][3][j] - I[i][4][j];
				T0[i][0][j] = t0+t1+O_trans_h[index8(i,0,j,d,h,w,n,k,l1,l2,l3,l4,l5,l6,l7)];//I[i][0][j];
				T0[i][1][j] = t2*0.700 + t3*1.500;
				T0[i][2][j] = t0*0.490 + t1*2.250;
				T0[i][3][j] = t2*0.343 + t3*3.375 + O_trans_h[index8(i,5,j,d,h,w,n,k,l1,l2,l3,l4,l5,l6,l7)];//}//I[i][5][j];
			}
	}
#endif
	for(int i=0;i<imTile;i++){
			#pragma unroll
			for(int j=0;j<outTile;j++){
				float t0 = T0[i][j][1] + T0[i][j][2];
				float t1 = T0[i][j][3] + T0[i][j][4];
				float t2 = T0[i][j][1] - T0[i][j][2];
				float t3 = T0[i][j][3] - T0[i][j][4];
				T1[i][j][0] = t0+t1+T0[i][j][0];
				T1[i][j][1] = t2*0.700 + t3*1.500;
				T1[i][j][2] = t0*0.490 + t1*2.250;
				T1[i][j][3] = t2*0.343 + t3*3.375 + T0[i][j][5];
			}
	}

#ifdef layout1
	for(int i=0;i<outTile;i++){
		#pragma unroll
		for(int j=0;j<outTile;j++){
			float t0 = T1[1][i][j] + T1[2][i][j];
			float t1 = T1[3][i][j] + T1[4][i][j];
			float t2 = T1[1][i][j] - T1[i][2][j];
			float t3 = T1[3][i][j] - T1[4][i][j];
			T2[0][i][j] = t0+t1+T1[0][i][j];
			T2[1][i][j] = t2*0.700 + t3*1.500;
			T2[2][i][j] = t0*0.490 + t1*2.250;
			T2[3][i][j] = t2*0.343 + t3*3.375 + T1[5][i][j];
			//if(valid_k){
			if((h*outTile+i<P)&&(w*outTile+j<Q)){
				if(d*outTile+3<M){
					O_winograd_h[index5(n,k,(d*outTile+0),(h*outTile+i),(w*outTile+j),KMPQ,MPQ,PQ,Q)] = T2[0][i][j];
					O_winograd_h[index5(n,k,(d*outTile+1),(h*outTile+i),(w*outTile+j),KMPQ,MPQ,PQ,Q)] = T2[1][i][j];
					O_winograd_h[index5(n,k,(d*outTile+2),(h*outTile+i),(w*outTile+j),KMPQ,MPQ,PQ,Q)] = T2[2][i][j];
					O_winograd_h[index5(n,k,(d*outTile+3),(h*outTile+i),(w*outTile+j),KMPQ,MPQ,PQ,Q)] = T2[3][i][j];
				}
				else if(d*outTile+2<M){
					O_winograd_h[index5(n,k,(d*outTile+0),(h*outTile+i),(w*outTile+j),KMPQ,MPQ,PQ,Q)] = T2[0][i][j];
					O_winograd_h[index5(n,k,(d*outTile+1),(h*outTile+i),(w*outTile+j),KMPQ,MPQ,PQ,Q)] = T2[1][i][j];
					O_winograd_h[index5(n,k,(d*outTile+2),(h*outTile+i),(w*outTile+j),KMPQ,MPQ,PQ,Q)] = T2[2][i][j];
				}
				else if(d*outTile+1<M){
					O_winograd_h[index5(n,k,(d*outTile+0),(h*outTile+i),(w*outTile+j),KMPQ,MPQ,PQ,Q)] = T2[0][i][j];
					O_winograd_h[index5(n,k,(d*outTile+1),(h*outTile+i),(w*outTile+j),KMPQ,MPQ,PQ,Q)] = T2[1][i][j];
				}
				else
					O_winograd_h[index5(n,k,(d*outTile+0),(h*outTile+i),(w*outTile+j),KMPQ,MPQ,PQ,Q)] = T2[0][i][j];
			}//}

		}
	}
#else

	int ll3 = N*K;
	int ll2 = Q*ll3;
	int ll1 = P*ll2;

	for(int i=0;i<outTile;i++){
		#pragma unroll
		for(int j=0;j<outTile;j++){
			float t0 = T1[1][i][j] + T1[2][i][j];
			float t1 = T1[3][i][j] + T1[4][i][j];
			float t2 = T1[1][i][j] - T1[i][2][j];
			float t3 = T1[3][i][j] - T1[4][i][j];
			T2[0][i][j] = t0+t1+T1[0][i][j];
			T2[1][i][j] = t2*0.700 + t3*1.500;
			T2[2][i][j] = t0*0.490 + t1*2.250;
			T2[3][i][j] = t2*0.343 + t3*3.375 + T1[5][i][j];
			//if(valid_k){
			if((h*outTile+i<P)&&(w*outTile+j<Q)){
				if(d*outTile+3<M){
					O_winograd_h[index5((d*outTile+0),(h*outTile+i),(w*outTile+j),n,k,ll1,ll2,ll3,K)] = T2[0][i][j];
					O_winograd_h[index5((d*outTile+1),(h*outTile+i),(w*outTile+j),n,k,ll1,ll2,ll3,K)] = T2[1][i][j];
					O_winograd_h[index5((d*outTile+2),(h*outTile+i),(w*outTile+j),n,k,ll1,ll2,ll3,K)] = T2[2][i][j];
					O_winograd_h[index5((d*outTile+3),(h*outTile+i),(w*outTile+j),n,k,ll1,ll2,ll3,K)] = T2[3][i][j];
				}
				else if(d*outTile+2<M){
					O_winograd_h[index5((d*outTile+0),(h*outTile+i),(w*outTile+j),n,k,ll1,ll2,ll3,K)] = T2[0][i][j];
					O_winograd_h[index5((d*outTile+1),(h*outTile+i),(w*outTile+j),n,k,ll1,ll2,ll3,K)] = T2[1][i][j];
					O_winograd_h[index5((d*outTile+2),(h*outTile+i),(w*outTile+j),n,k,ll1,ll2,ll3,K)] = T2[2][i][j];
				}
				else if(d*outTile+1<M){
					O_winograd_h[index5((d*outTile+0),(h*outTile+i),(w*outTile+j),n,k,ll1,ll2,ll3,K)] = T2[0][i][j];
					O_winograd_h[index5((d*outTile+1),(h*outTile+i),(w*outTile+j),n,k,ll1,ll2,ll3,K)] = T2[1][i][j];
				}
				else
					O_winograd_h[index5((d*outTile+0),(h*outTile+i),(w*outTile+j),n,k,ll1,ll2,ll3,K)] = T2[0][i][j];
			}//}
		}
	}
#endif
}



////////////////////////////////////////////////////////////////////////////////
// Program main
//version 1: winograd algorithn imnplementation on GPU,
//there are six kernels, to do six tasks, filter transform , reshape transformed filter, image transform, reshape transformed image, reshape output, transform output.
//data layout:O_h[N,K,M,P,Q]  I_h[N,C,D,H,W]  K_h[K,C,T,R,S]

//version 2: new data layout: O_h[K,M,P,Q,N]  I_h[C,D,H,W,N]  K_h[C,T,R,S,K]
////////////////////////////////////////////////////////////////////////////////

//version 3: based on version 2 and integrate reshape kernel to transform kernels.
int
main(int argc, char **argv)
{
    runTest(argc, argv);
}

////////////////////////////////////////////////////////////////////////////////
//! Run a simple test for CUDA
////////////////////////////////////////////////////////////////////////////////
void
runTest(int argc, char **argv)
{
	//Initial input
	int N = 32;
	int C = 64;//16;
	int K = 128;//32;
	int D = 16;//64;
	int H = 16;//64;
	int W = 16;//64;

	int pad = 0;
	int pad_d = pad;
	int pad_h = pad;
	int pad_w = pad;
	//int stride_d = 1;
	//int stride_h = 1;
	//int stride_w = 1;
	int T = 3;
	int R = 3;
	int S = 3;

	int M = D+2*pad_d-T+1;
	int P = H+2*pad_h-R+1;
	int Q = W+2*pad_w-S+1;

	int a1 = 1;
		int a2 = 2;
		int a3 = 3;
		int a4 = 4;
		int a5 = 5;
		int test = index5((a1+a2),a3,a4,a5,(a1+a3),2,3,4,5);
		printf("test = %d,\n",test);

	int data_num = C*D*H*W*N;
	//allocate Input Image data on host
	float *I_h = (float *)malloc(sizeof(float)*data_num);
	for(int i=0;i<data_num;i++)
		I_h[i] = (rand()/float(RAND_MAX));

	int kernel_num = C*K*T*R*S;
	float *K_h = (float *)malloc(sizeof(float)*kernel_num);

	printf("K_h value is :\n");
	for(int i=0;i<kernel_num;i++){
		K_h[i] = (rand()/float(RAND_MAX));
		//printf("K_h[%d] = %g\n",i,K_h[i]);
	}

	int O_num = K*M*P*Q*N;
	float *O_h = (float *)malloc(sizeof(float)*O_num);
	float *O_H = (float *)malloc(sizeof(float)*O_num);
	for(int i=0;i<O_num;i++){
		O_h[i] = 0.0;
		O_H[i] = 0.0;
	}

	float *I_H = (float *)malloc(sizeof(float)*data_num);
	float *K_H = (float *)malloc(sizeof(float)*kernel_num);

	for(int c=0;c<C;c++)
		for(int d=0;d<D;d++)
			for(int h=0;h<H;h++)
				for(int w=0;w<W;w++)
					for(int n=0;n<N;n++)
						I_H[index5(c,d,h,w,n,D*H*W*N,H*W*N,W*N,N)] = I_h[index5(n,c,d,h,w,C*D*H*W,D*H*W,H*W,W)];
	for(int c=0;c<C;c++)
		for(int t=0;t<T;t++)
			for(int r=0;r<R;r++)
				for(int s=0;s<S;s++)
					for(int k=0;k<K;k++)
						K_H[index5(c,t,r,s,k,T*R*S*K,R*S*K,S*K,K)] = K_h[index5(k,c,t,r,s,C*T*R*S,T*R*S,R*S,S)];


	int gpuid = 0;
	checkCudaErrors(cudaSetDevice(gpuid));
	float *I_d,*K_d,*O_d;
	float *Ocudnn_h = (float *)malloc(sizeof(float)*O_num);
	checkCudaErrors(cudaMalloc(&I_d, sizeof(float)*data_num));
	checkCudaErrors(cudaMalloc(&K_d, sizeof(float)*kernel_num));
	checkCudaErrors(cudaMalloc(&O_d, sizeof(float)*O_num));
	checkCudaErrors(cudaMemcpy(I_d, &I_h[0], sizeof(float)*data_num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(K_d, &K_h[0], sizeof(float)*kernel_num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(O_d, &O_h[0], sizeof(float)*O_num, cudaMemcpyHostToDevice));

	float *I_D,*K_D;
	checkCudaErrors(cudaMalloc(&I_D, sizeof(float)*data_num));
	checkCudaErrors(cudaMalloc(&K_D, sizeof(float)*kernel_num));
	checkCudaErrors(cudaMemcpy(I_D, &I_H[0], sizeof(float)*data_num, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(K_D, &K_H[0], sizeof(float)*kernel_num, cudaMemcpyHostToDevice));


	printf("computing started!\n");
	double t1 = timing();
	double  floatOperations = ((N*K*M*P*Q*C)/(1e+9))*T*R*S*2;
	//
	/*for(int i=0;i<N;i++)
		for(int k=0;k<K;k++)
			for(int m=0;m<M;m++)
				for(int p=0;p<P;p++)
					for(int q=0;q<Q;q++)
						for(int j=0;j<C;j++)
							for(int t=0;t<T;t++)
								for(int r=0;r<R;r++)
									for(int s=0;s<S;s++) {
										int d,h,w;
										d = m-pad_d;
										h = p-pad_h;
										w = q-pad_w;
										if((d>=0) && (h>=0) && (w>=0))
											//O_h[k][m][p][q][i]+=I_h[j][d+t][h+r][w+s][i]*K_h[j][t][r][s][k];
											//O_h[i][m][p][q][k]+=I_h[i][d+t][h+r][w+s][j]*K_h[k][t][r][s][j];
											//O_h[i*M*P*Q*K+m*P*Q*K+p*Q*K+q*K+k] += I_h[i*D*H*W*C+(d+t)*H*W*C+(h+r)*W*C+(w+s)*C+j]*\
											//									  K_h[k*T*R*S*C+t*R*S*C+r*S*C+s*C+j];
											//O_h[i*M*P*Q*K+k*P*Q*M+m*Q*P+p*Q+q] += I_h[i*D*H*W*C+j*D*H*W+(d+t)*H*W+(h+r)*W+(w+s)]*\
											//									  K_h[k*T*R*S*C+j*T*R*S+t*R*S+r*S+s];
											//O_h[N,K,M,P,Q]  I_h[N,C,D,H,W]  K_h[K,C,T,R,S]
										    //O_h[index5(i,k,m,p,q,M*P*Q*K,M*P*Q,P*Q,Q)]+=I_h[index5(i,j,(d+t),(h+r),(w+s),C*D*H*W,D*H*W,H*W,W)]*K_h[index5(k,j,t,r,s,C*T*R*S,T*R*S,R*S,S)];
											//O_h[K,M,P,Q,N]  I_h[C,D,H,W,N]  K_h[C,T,R,S,K]
											O_H[index5(k,m,p,q,i,M*P*Q*N,P*Q*N,Q*N,N)]+=I_H[index5(j,(d+t),(h+r),(w+s),i,D*H*W*N,H*W*N,W*N,N)]*K_H[index5(j,t,r,s,k,T*R*S*K,R*S*K,S*K,K)];

									}*/
	printf("computing finished1!\n");
	double t2 = timing();
	printf("Time for 3D convolution using direct method is :%g\n",t2-t1);



	//kernel transform F(2x2x2,3x3x3)
	int fm = 4;
	int fk = 3;
	int fmk = fm+fk-1;
	float G[4*3] = {1, 		0, 		0,
					0.5, 	0.5, 	0.5,
					0.5, 	-0.5, 	0.5,
					0, 		0, 		1};
	int ktileSize = fmk;


	float *F_trans_d;
	int ftransNum = K*C*ktileSize*ktileSize*ktileSize;
	checkCudaErrors(cudaMalloc(&F_trans_d, sizeof(float)*ftransNum));
#ifdef DEBUG
	float *F_trans_h = (float *)malloc(sizeof(float)*ftransNum);
#endif
	// Setup execution parameters
	dim3 threads(32,1,1);
	int K32 = ceil(((float)K)/32.0);
	dim3 grid(K32,C,1);

	//image transform on CPU
	float *I_trans_h;
	int imTile = fmk;
	int outTile = fm;
	int dTileNum = ceil((float)M/(float)outTile);
	int hTileNum = ceil((float)P/(float)outTile);
	int wTileNum = ceil((float)Q/(float)outTile);

	int imTile3 = imTile*imTile*imTile;
	int imTile2 = imTile*imTile;
	int dhwTileNum = dTileNum*hTileNum*wTileNum;
	int hwTileNum = hTileNum*wTileNum;

	int imtransSize = N*C*dTileNum*hTileNum*wTileNum*imTile*imTile*imTile;

	//image transform on GPU
	float *I_trans_d;
	checkCudaErrors(cudaMalloc(&I_trans_d, sizeof(float)*imtransSize));
	dim3 ImTransThreads(32,1,1);
	int GN = ceil(((float)N)/32.0);
	int GX = dTileNum;
	int GY = hTileNum;
	int GZ = wTileNum;
	int GXYZ = GX*GY*GZ;
	dim3 ImTransGrid(GN,GXYZ,C);



	int outSize = N*K*dTileNum*hTileNum*wTileNum*imTile*imTile*imTile;
	float *O_trans_h = (float *)malloc(sizeof(float)*outSize);
	//float *O_trans_d;
	int dotFSize = C*K;
	float *F_temp_h = (float *)malloc(sizeof(float)*dotFSize);
	float *F_temp_H = (float *)malloc(sizeof(float)*dotFSize);
	float *F_temp_d;
	checkCudaErrors(cudaMalloc(&F_temp_d, sizeof(float)*dotFSize));
	int dotImSize = C*dTileNum*hTileNum*wTileNum*N;
	float *Im_temp_h = (float *)malloc(sizeof(float)*dotImSize);
	float *Im_temp_d;
	checkCudaErrors(cudaMalloc(&Im_temp_d, sizeof(float)*dotImSize));
	float *O_temp_d;
	int dotOSize = K*N*dTileNum*hTileNum*wTileNum;
	checkCudaErrors(cudaMalloc(&O_temp_d, sizeof(float)*dotOSize));
	float *O_temp_h = (float *)malloc(sizeof(float)*dotOSize);

	float *O_trans_d;
	checkCudaErrors(cudaMalloc(&O_trans_d, sizeof(float)*outSize));
	//checkCudaErrors(cudaMemcpy(O_trans_d, &O_trans_h[0], sizeof(float)*outSize, cudaMemcpyHostToDevice));

	float *O_winograd_d;
	checkCudaErrors(cudaMalloc(&O_winograd_d, sizeof(float)*O_num));

	cublasHandle_t cublasHandle;
	checkCudaErrors(cublasCreate(&cublasHandle));

	dim3 ReshapeFilterThreads(32,1,1);
	dim3 ReshapeFilterGrids(K32,C,1);

	dim3 OutTransThreads(32,1,1);
	// GN = ceil(((float)N)/32.0);
	//int GX = dTileNum;
	//int GY = hTileNum;
	//int GZ = wTileNum;
	//int GXYZ = GX*GY*GZ;
	dim3 OutTransGrid(GN,GXYZ,K);

	dim3 OutTransThreadsNew(32,1,1);
	int GK = ceil(((float)K)/32.0);
	dim3 OutTransGridNew(GK,GXYZ,N);

	int CimTile3 = C*imTile3;
	printf("CimTile3=%d\n",CimTile3);

	double reshapeTransFilterCPUT=0,reshapeImageTransCPUT=0,shapedFilerTransferT=0,shapedImageTransferT=0,dotOperationT=0,reshapeOutCPUT=0,transferOutbackT=0;
	double reshapeTransFilterGPUT=0;
	double reshapeTransImageGPUT=0;
	double reshapeOutGPUT = 0;

	int KM = N*dTileNum*hTileNum*wTileNum;
	const float alpha = 1.0;
	const float beta = 0.0;
#ifdef layout1
	int lda = K;//C;
	int ldb = C;
	int ldc = K;
#else
	int lda = K;
	int ldb = KM;
	int ldc = K;
#endif
	double subfloatOperations = (double(K*C*KM*2)/1000.0);

	cudaEvent_t start, stop;
	checkCudaErrors(cudaEventCreate(&start));
	checkCudaErrors(cudaEventCreate(&stop));

	checkCudaErrors(cudaEventRecord(start, NULL));

	double GPU_start = timing();
	double filtertransT1 = timing();
#ifdef layout1
	FilterTransform_4x4x4_3x3x3<<<grid,threads>>>(K_d,K,C,T,R,S,ktileSize,F_trans_d);
#else
	FilterTransform_4x4x4_3x3x3<<<grid,threads>>>(K_D,K,C,T,R,S,ktileSize,F_trans_d);
#endif
	double filtertransT2 = timing();
	printf("filter transforming time is on GPU is %g\n",filtertransT2-filtertransT1);

#ifdef DEBUG
	checkCudaErrors(cudaMemcpy(F_trans_h, F_trans_d, sizeof(float)*ftransNum, cudaMemcpyDeviceToHost));
	printf("tansformed filter is :\n");
	for(int i=0;i<ftransNum;i++)
		printf("%d %g\n",i,F_trans_h[i]);
#endif
	double ImageTransformGPUT1 = timing();
#ifdef layout1
	ImageTransform_4x4x4_3x3x3_d<<<ImTransGrid,ImTransThreads>>>(I_d, I_trans_d,imTile,outTile,dTileNum,hTileNum,wTileNum,pad_d,pad_h,pad_w, N, C, D, H, W);
#else
	ImageTransform_4x4x4_3x3x3_d<<<ImTransGrid,ImTransThreads>>>(I_D, I_trans_d,imTile,outTile,dTileNum,hTileNum,wTileNum,pad_d,pad_h,pad_w, N, C, D, H, W);
#endif
	double ImageTransformGPUT2 = timing();
	printf("time for image transform on GPU is %g\n",ImageTransformGPUT2-ImageTransformGPUT1);

	//dot operation
	for(int r=0;r<imTile;r++)
		for(int s=0;s<imTile;s++)
			for(int t=0;t<imTile;t++){

				//reshape transformed filter on GPU
				double FtempGPUT = timing();
				//ReshapeTransformedFilterOnGPU<<<ReshapeFilterGrids,ReshapeFilterThreads>>>(F_temp_d,F_trans_d,K,C,imTile,r,s,t,imTile3, CimTile3, imTile2);
				reshapeTransFilterGPUT+=timing()-FtempGPUT;


				//reshape transformed image on GPU
				double ImtempGPUT = timing();
				//ReshapeTransformedImageOnGPU<<<ImTransGrid,ImTransThreads>>>(Im_temp_d, I_trans_d, C, N, dhwTileNum, hwTileNum, dTileNum, hTileNum, wTileNum, CimTile3, imTile3, imTile2, imTile, r, s, t);
				reshapeTransImageGPUT+=timing()-ImtempGPUT;

				//call cublas to do dot operation
				double dotOperationStart = timing();
#ifdef layout1
				checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_N, K, KM, C, &alpha, F_trans_d+(r*imTile*imTile+s*imTile+t)*K*C,
				                                   lda, I_trans_d+(r*imTile*imTile+s*imTile+t)*KM*C, ldb, &beta, O_trans_d+(r*imTile*imTile+s*imTile+t)*KM*K, ldc));//lda = K
				//checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_T, CUBLAS_OP_N, K, KM, C, &alpha, F_temp_d,
				//                                     lda, Im_temp_d, ldb, &beta, O_temp_d, ldc));//lda = C
#else
				//F_temp_d: [K C]  Im_temp_d: [KM C]  , so Im_temp_d need to transpose
				checkCudaErrors(cublasSgemm(cublasHandle, CUBLAS_OP_N, CUBLAS_OP_T, K, KM, C, &alpha, F_trans_d+(r*imTile*imTile+s*imTile+t)*K*C,
								                                   lda, I_trans_d+(r*imTile*imTile+s*imTile+t)*KM*C, ldb, &beta, O_trans_d+(r*imTile*imTile+s*imTile+t)*KM*K, ldc));//lda = K
#endif
				double subdotOperationT = timing()-dotOperationStart;
				dotOperationT += subdotOperationT;

				printf("dot production for each call is %g float performance is %g\n",subdotOperationT,subfloatOperations/subdotOperationT/(1000000.0));

				//reshape transformed out on GPU
				double reshapeResultGPUStart = timing();
				//ReshapeOutOnGPU<<<OutTransGrid,OutTransThreads>>>(O_trans_d, O_temp_d, K, N, dhwTileNum, hwTileNum, dTileNum, hTileNum, wTileNum, imTile3, imTile2, imTile, r, s, t);
				reshapeOutGPUT += timing()-reshapeResultGPUStart;

			}


	double OutTransformGPUT1 = timing();
	OutputTransform_4x4x4_3x3x3_d<<<OutTransGridNew,OutTransThreadsNew>>>(O_winograd_d, O_trans_d,imTile,outTile,dTileNum,hTileNum,wTileNum, N, K, M, P, Q);
	double OutTransformGPUT2 = timing();

	printf("Time information: \n");
	printf("dotOperationT=%g, \n", dotOperationT);
	printf("reshapeTransFilterGPUT = %g\n",reshapeTransFilterGPUT);
	printf("reshapeTransImageGPUT = %g\n",reshapeTransImageGPUT);
	printf("reshapeOutGPUT = %g\n",reshapeOutGPUT);

	printf("time for out transform on GPU is %g\n",OutTransformGPUT2-OutTransformGPUT1);

	double GPU_end = timing();
	printf("winograd alogithm on GPU total time is %g\n",GPU_end-GPU_start);
	checkCudaErrors(cudaEventRecord(stop, NULL));
	checkCudaErrors(cudaEventSynchronize(stop));
	float msecTotal = 0.0f;
	checkCudaErrors(cudaEventElapsedTime(&msecTotal, start, stop));
	printf("winograd alogithm on GPU total time  using event to record is %g\n",msecTotal/1000.0);


	float *Out_trans = (float *)malloc(sizeof(float)*O_num);
	double OuttransTransferT1 = timing();
	checkCudaErrors(cudaMemcpy(Out_trans, O_winograd_d, sizeof(float)*O_num, cudaMemcpyDeviceToHost));
	double OuttransTransferT2 = timing();
	printf("time for transfering Out transformed data is %g\n",OuttransTransferT2-OuttransTransferT1);

	double cublasOperations = ((double(N*dTileNum*hTileNum*wTileNum*K*C))/1000.0)*imTile*imTile*imTile*2;
	printf("cublas performance is %g\n",cublasOperations/dotOperationT/(1000000.0));

	printf("M=%d P=%d Q=%d imTile=%d outTile=%d dTileNum=%d hTileNum=%d wTileNum=%d \n",M,P,Q,imTile,outTile,dTileNum,hTileNum,wTileNum);
	//cudnn method
	// Create CUBLAS and CUDNN handles
	// Initialize CUDNN/CUBLAS training context
	cudnnHandle_t cudnnHandle;

	cudnnTensorDescriptor_t dataTensor, conv1Tensor;
	cudnnFilterDescriptor_t filterDesc;
	cudnnConvolutionDescriptor_t convDesc;
	cudnnConvolutionFwdAlgo_t conv1algo;

	int m_gpuid=gpuid;

	int m_batchSize;
	size_t m_workspaceSize;

	int batch_size = N;
	m_batchSize = batch_size;

	checkCUDNN(cudnnCreate(&cudnnHandle));

	// Create tensor descriptors
	checkCUDNN(cudnnCreateTensorDescriptor(&dataTensor));
	checkCUDNN(cudnnCreateTensorDescriptor(&conv1Tensor));

	checkCUDNN(cudnnCreateFilterDescriptor(&filterDesc));
	checkCUDNN(cudnnCreateConvolutionDescriptor(&convDesc));

	ConvBiasLayer conv1(C, K, R, H, W);


	// Set convolution tensor sizes and compute workspace size
	size_t  workspace = 0;
	//size_t workspace1 = SetFwdConvolutionTensors(cudnnHandle, conv1, m_batchSize, dataTensor, conv1Tensor, conv1filterDesc, conv1Desc, conv1algo);
	int ndims = 5;
	int *imageShape = (int *)malloc(sizeof(int)*ndims);
	int *filterShape = (int *)malloc(sizeof(int)*ndims);
	int *outShape = (int *)malloc(sizeof(int)*ndims);

	int *imageStride = (int *)malloc(sizeof(int)*ndims);
	int *filterStride = (int *)malloc(sizeof(int)*ndims);
	int *outStride = (int *)malloc(sizeof(int)*ndims);

	imageStride[ndims-1] = 1;
	filterStride[ndims-1] = 1;
	outStride[ndims-1] = 1;

	imageShape[0] = N;		imageShape[1] = C;		imageShape[2] = D;
	imageShape[3] = H;		imageShape[4] = W;

	outShape[0] = N;		outShape[1] = K;		outShape[2] = M;
	outShape[3] = P;		outShape[4] = Q;

	filterShape[0] = K;		filterShape[1] = C;		filterShape[2] = T;
	filterShape[3] = R;		filterShape[4] = S;

	for(int i=ndims-2;i>=0;i--){
		imageStride[i] = imageStride[i+1]*imageShape[i+1];
		filterStride[i] = filterStride[i+1]*filterShape[i+1];
	}


	checkCUDNN(cudnnSetTensorNdDescriptor(dataTensor,
										  	  CUDNN_DATA_FLOAT,
										  	  ndims,
										  	  imageShape,
										  	  imageStride));
	if(dataTensor==NULL)
		printf("datatensor null\n");



	checkCUDNN(cudnnSetFilterNdDescriptor(filterDesc,
												CUDNN_DATA_FLOAT,
												CUDNN_TENSOR_NCHW,
												ndims,
												filterShape));
	if(filterDesc==NULL)
			printf("filterDesc null\n");

	int *padArray = (int *)malloc((ndims-2)*sizeof(int));
	int *strideArray = (int *)malloc((ndims-2)*sizeof(int));
	int *upscale = (int *)malloc((ndims-2)*sizeof(int));
	for(int i=0;i<(ndims-2);i++) {
		padArray[i] = pad;
		strideArray[i] = 1;
		upscale[i] = 1;
	}

	checkCUDNN(cudnnSetConvolutionNdDescriptor(convDesc,
												ndims-2,
												padArray,
												strideArray,
												upscale,
												CUDNN_CROSS_CORRELATION,
												CUDNN_DATA_FLOAT));
	if(convDesc==NULL)
			printf("convDesc null\n");

	// Find dimension of convolution output
	checkCUDNN(cudnnGetConvolutionNdForwardOutputDim(convDesc,
														dataTensor,
														filterDesc,
														ndims,
														outShape));

	for(int i=ndims-2;i>=0;i--){
		outStride[i] = outStride[i+1]*outShape[i+1];
	}

	checkCUDNN(cudnnSetTensorNdDescriptor(conv1Tensor,
												 CUDNN_DATA_FLOAT,
												 ndims,
												 outShape,
												 outStride));
	for(int i=0;i<ndims;i++)
		printf("outshape[%d] = %d \n",i,outShape[i]);
	checkCUDNN(cudnnGetConvolutionForwardAlgorithm(cudnnHandle,
														dataTensor,
														filterDesc,
														convDesc,
														conv1Tensor,
														CUDNN_CONVOLUTION_FWD_PREFER_FASTEST,
														0,
														&conv1algo));

	//conv1algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
	//conv1algo = CUDNN_CONVOLUTION_FWD_ALGO_GEMM;
	//printf("conv algorithm is %d \n",conv1algo);
	checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnnHandle,
															   dataTensor,
	                                                           filterDesc,
	                                                           convDesc,
	                                                           conv1Tensor,
	                                                           conv1algo,
	                                                           &workspace));

	workspace = std::max(workspace, workspace);


	// The workspace is allocated later (if necessary)
	m_workspaceSize = workspace;
	printf("workspace  size is %d \n",(int)workspace);
	void *d_cudnn_workspace = nullptr;
	if (m_workspaceSize > 0)
	    checkCudaErrors(cudaMalloc(&d_cudnn_workspace, m_workspaceSize));
	//float alpha = 1.0f, beta = 0.0f;
	checkCudaErrors(cudaSetDevice(m_gpuid));
	double cudnnConvolutionStart = timing();
	// Conv1 layer
	checkCUDNN(cudnnConvolutionForward(cudnnHandle, &alpha, dataTensor,
	                                                   I_d, filterDesc, K_d, convDesc,
	                                                   conv1algo, d_cudnn_workspace, m_workspaceSize, &beta,
	                                                   conv1Tensor, O_d));


	double cudnnResultTransferStart =  timing();
	checkCudaErrors(cudaMemcpy(Ocudnn_h, O_d, sizeof(float)*O_num, cudaMemcpyDeviceToHost));
	double cudnnResultTransferT = timing()-cudnnResultTransferStart;
	printf("time for cudnn result transfering is %g \n", cudnnResultTransferT);

	double cudnnConvolutionT = timing()-cudnnConvolutionStart;
	printf("time for convolution using cudnn is %g and float performance is %g\n",cudnnConvolutionT,floatOperations/(cudnnConvolutionT-cudnnResultTransferT));


	printf("Checking computed result for correctness:\n ");

	double eps = 1.e-5 ; // machine zero

	for(int k=0;k<K;k++)
		for(int m=0;m<M;m++)
			for(int p=0;p<P;p++)
				for(int q=0;q<Q;q++)
					for(int i=0;i<N;i++)
						O_h[index5(i,k,m,p,q,K*M*P*Q,M*P*Q,P*Q,Q)] = O_H[index5(k,m,p,q,i,M*P*Q*N,P*Q*N,Q*N,N)];

	bool resCUBLAS;
    //resCUBLAS = sdkCompareL2fe(O_h, Ocudnn_h, O_num,eps);
	//printf("Comparing cudnn result with CPU results: %s\n", (true == resCUBLAS) ? "PASS" : "FAIL");
#ifdef layout1
	resCUBLAS = sdkCompareL2fe(Ocudnn_h, Out_trans, O_num,eps);
	printf("NCDHW format: Comparing winograd result withcudnn results: %s\n", (true == resCUBLAS) ? "PASS" : "FAIL");
	for(int i=0;i<128;i++)
			printf("Ocudnn_h[%d] = %g  Out_trans[%d]=%g \n",i,Ocudnn_h[i],i,Out_trans[i]);

#else
	float *Out_trans_h = (float *)malloc(sizeof(float)*O_num);
	for(int k=0;k<K;k++)
		for(int m=0;m<M;m++)
			for(int p=0;p<P;p++)
				for(int q=0;q<Q;q++)
					for(int i=0;i<N;i++)
						Out_trans_h[index5(i,k,m,p,q,K*M*P*Q,M*P*Q,P*Q,Q)] = Out_trans[index5(m,p,q,i,k,P*Q*N*K,Q*N*K,N*K,K)];//Out_trans[index5(k,m,p,q,i,M*P*Q*N,P*Q*N,Q*N,N)];

	resCUBLAS = sdkCompareL2fe(Ocudnn_h, Out_trans_h, O_num,eps);
	printf("CDHWN format: Comparing winograd result withcudnn results: %s\n", (true == resCUBLAS) ? "PASS" : "FAIL");
	for(int i=0;i<128;i++)
		printf("Ocudnn_h[%d] = %g  Out_trans_h[%d]=%g \n",i,Ocudnn_h[i],i,Out_trans_h[i]);
	free(Out_trans_h);
#endif

	printf("computing finished2!\n");

	free(I_h);
	free(K_h);
	free(I_H);
	free(K_H);
	free(O_h);


	free(I_trans_h);
	free(O_trans_h);
	free(F_temp_h);
	free(Im_temp_h);
	free(O_temp_h);
	free(imageShape);
	free(filterShape);
	free(outShape);
	free(imageStride);
	free(filterStride);
	free(outStride);
	free(Out_trans);


	free(Ocudnn_h);

	checkCudaErrors(cudaFree(I_d));
	checkCudaErrors(cudaFree(K_d));
	checkCudaErrors(cudaFree(I_D));
	checkCudaErrors(cudaFree(K_D));
	checkCudaErrors(cudaFree(O_d));
	checkCudaErrors(cudaFree(F_trans_d));
	checkCudaErrors(cudaFree(I_trans_d));
	checkCudaErrors(cudaFree(O_winograd_d));

	checkCudaErrors(cudaFree(O_trans_d));
	checkCudaErrors(cudaFree(F_temp_d));
	checkCudaErrors(cudaFree(O_temp_d));
	checkCudaErrors(cudaFree(Im_temp_d));
	if(d_cudnn_workspace!=nullptr)
		checkCudaErrors(cudaFree(d_cudnn_workspace));

	checkCudaErrors(cudaSetDevice(m_gpuid));
	checkCUDNN(cudnnDestroy(cudnnHandle));
	checkCUDNN(cudnnDestroyTensorDescriptor(dataTensor));
	checkCUDNN(cudnnDestroyTensorDescriptor(conv1Tensor));
	checkCUDNN(cudnnDestroyFilterDescriptor(filterDesc));
	checkCUDNN(cudnnDestroyConvolutionDescriptor(convDesc));



}
