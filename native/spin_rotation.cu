// SPDX-License-Identifier: GPL-3.0-or-later
#include "spin_rotation.hpp"
#include <cuComplex.h>
#include <cmath>
#include <cstdio>

namespace {
struct Rotation {cuDoubleComplex u[4];};
double previous[4]={1.,0.,0.,0.};
Rotation pending{};
int selected=0;
bool applied=true;
bool atomic_guess=false;
__global__ void rotate_density(int capacity,int spins,double* d,Rotation r) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=capacity)return;
    const auto a=make_cuDoubleComplex(d[i],spins==8?d[i+4*capacity]:0.);
    const auto b=make_cuDoubleComplex(d[i+capacity],spins==8?d[i+5*capacity]:0.);
    const auto c=make_cuDoubleComplex(d[i+2*capacity],-d[i+3*capacity]);
    const auto f=spins==8?make_cuDoubleComplex(d[i+6*capacity],d[i+7*capacity]):cuConj(c);
    const cuDoubleComplex matrix[4]={a,c,f,b};
    cuDoubleComplex left[4],out[4];
    #pragma unroll
    for(int row=0;row<2;row++)for(int col=0;col<2;col++)
        left[2*row+col]=cuCadd(cuCmul(r.u[2*row],matrix[col]),cuCmul(r.u[2*row+1],matrix[2+col]));
    #pragma unroll
    for(int row=0;row<2;row++)for(int col=0;col<2;col++)
        out[2*row+col]=cuCadd(cuCmul(left[2*row],cuConj(r.u[2*col])),cuCmul(left[2*row+1],cuConj(r.u[2*col+1])));
    d[i]=out[0].x;d[i+capacity]=out[3].x;
    d[i+2*capacity]=out[1].x;d[i+3*capacity]=-out[1].y;
    if(spins==8){
        d[i+4*capacity]=out[0].y;d[i+5*capacity]=out[3].y;
        d[i+6*capacity]=out[2].x;d[i+7*capacity]=out[2].y;
    }
}
}
extern "C" int remag_orientation_set(int direction,const double* quaternion) {
    if(direction!=selected+1||!applied||!quaternion)return -220;
    double q[4],norm=0.;
    for(int k=0;k<4;k++){
        if(!std::isfinite(quaternion[k]))return -220;
        norm+=quaternion[k]*quaternion[k];
    }
    if(!std::isfinite(norm)||norm<1.e-24)return -220;
    norm=std::sqrt(norm);for(int k=0;k<4;k++)q[k]=quaternion[k]/norm;
    // Absolute requested rotation times inverse(previous requested rotation).
    const double p[4]={atomic_guess?1.:previous[0],atomic_guess?0.:-previous[1],atomic_guess?0.:-previous[2],atomic_guess?0.:-previous[3]};
    const double w=q[0]*p[0]-q[1]*p[1]-q[2]*p[2]-q[3]*p[3];
    const double x=q[0]*p[1]+q[1]*p[0]+q[2]*p[3]-q[3]*p[2];
    const double y=q[0]*p[2]-q[1]*p[3]+q[2]*p[0]+q[3]*p[1];
    const double z=q[0]*p[3]+q[1]*p[2]-q[2]*p[1]+q[3]*p[0];
    // U = w I - i (x sigma_x + y sigma_y + z sigma_z).
    pending={{make_cuDoubleComplex(w,-z),make_cuDoubleComplex(-y,-x),
              make_cuDoubleComplex(y,-x),make_cuDoubleComplex(w,z)}};
    for(int k=0;k<4;k++)previous[k]=q[k];
    selected=direction;applied=false;return 0;
}
namespace remag {
int orientation_density(int capacity,int spins,double* density,cudaStream_t stream) {
    if(!selected)return 0;
    if(applied||capacity<1||(spins!=4&&spins!=8)||!density||!stream)return -220;
    rotate_density<<<(capacity+255)/256,256,0,stream>>>(capacity,spins,density,pending);
    const auto status=cudaGetLastError();
    if(status!=cudaSuccess)return static_cast<int>(status);
    applied=true;
    std::printf("REMAG GPU spin rotation: direction %d, %d sparse entries, %d density components\n",selected,capacity,spins);
    return 0;
}
}

namespace {bool retain_geometry=false;}
extern "C" int remag_orientation_cache_mode(int enabled,int atomic) {
    if(selected||(enabled!=0&&enabled!=1)||(atomic!=0&&atomic!=1))return -220;
    retain_geometry=enabled;atomic_guess=atomic;return 0;
}
extern "C" int remag_spin_constraint_retain_geometry();
// Fixed-geometry producers keep their state for a later orientation or a
// later spin-constraint trial at the same crystal geometry.
extern "C" int remag_orientation_retain_geometry() {
    return (retain_geometry&&selected>0)||remag_spin_constraint_retain_geometry();
}
