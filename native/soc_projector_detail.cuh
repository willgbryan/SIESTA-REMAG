// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include <cmath>
namespace remag::soc_detail {
struct Complex {double real,imag;};
__device__ inline Complex scaled(Complex a,double b){return {a.real*b,a.imag*b};}
__device__ inline Complex product_conjugate(Complex a,Complex b){return {a.real*b.real+a.imag*b.imag,a.imag*b.real-a.real*b.imag};}
__device__ inline void add(Complex& a,Complex b){a.real+=b.real;a.imag+=b.imag;}
__device__ inline Complex harmonic(const double* values,int l,int m,int stride) {
    if(abs(m)>l)return {0.,0.};
    if(m==0)return {values[l*stride],0.};
    const double fac=1./sqrt(2.);
    if(m<0)return {values[(l-m)*stride]*fac,values[(l+m)*stride]*fac};
    const double sign=m%2 ? -fac:fac;
    return {values[(l+m)*stride]*sign,-values[(l-m)*stride]*sign};
}
// Both contiguous oracle data and interleaved value/gradient records use this
// exact arithmetic. Status ownership belongs to the caller.
template<bool Gradients>
__device__ inline int evaluate_impl(int l,int split,const double* epsilon,const double* left,const double* right,
        const double* gradient,int value_stride,int gradient_stride,double* out) {
    const int width=2*l+1;
    for(int k=0;k<33;k++)out[k]=0.;
    for(int j=0;j<2;j++)if(!isfinite(epsilon[j]))return 2;
    for(int k=0;k<2*width;k++) {
        if(!isfinite(left[k*value_stride])||!isfinite(right[k*value_stride]))return 2;
        if constexpr(Gradients) for(int axis=0;axis<3;axis++)if(!isfinite(gradient[k*gradient_stride+axis]))return 2;
    }
    Complex v[4]={},f[12]={};
    for(int j=0;j<2;j++) {
        const int twice_j=2*l+2*j-1;
        for(int twice_mj=-twice_j;twice_mj<=twice_j;twice_mj+=2) {
            const double mj=0.5*twice_mj;
            const double cp=sqrt((l+0.5+mj)/width),cm=sqrt((l+0.5-mj)/width);
            const double cg[2]={j ? cp:-cm,j ? cm:cp};
            Complex a[2],b[2],g[6];
            for(int spin=0;spin<2;spin++) {
                const int m=(twice_mj+(spin ? 1:-1))/2;
                a[spin]=scaled(harmonic(left+j*width*value_stride,l,m,value_stride),cg[spin]);
                b[spin]=scaled(harmonic(right+j*width*value_stride,l,m,value_stride),cg[spin]);
                if constexpr(Gradients) for(int axis=0;axis<3;axis++)g[axis+3*spin]=scaled(harmonic(gradient+gradient_stride*j*width+axis,l,m,gradient_stride),cg[spin]);
            }
            for(int col=0;col<2;col++)for(int row=0;row<2;row++) {
                add(v[row+2*col],product_conjugate(scaled(a[row],epsilon[j]),b[col]));
                if constexpr(Gradients) for(int axis=0;axis<3;axis++)add(f[axis+3*row+6*col],product_conjugate(scaled(g[axis+3*row],epsilon[j]),b[col]));
            }
        }
    }
    double ion=0.;
    // Preserve the upstream guard on its full spin operator before splitting.
    if(hypot(v[2].real+v[1].real,v[2].imag-v[1].imag)>1e-4){return 3;}
    if(split) {
        const double product=epsilon[0]*epsilon[1];
        double mixed=sqrt(fabs(product));
        mixed*=product>0. ? copysign(1.,epsilon[0]):-1.;
        for(int m=0;m<width;m++) {
            const double term=(l*l*left[m*value_stride]*epsilon[0]*right[m*value_stride]
                +(l+1)*(l+1)*left[(width+m)*value_stride]*epsilon[1]*right[(width+m)*value_stride]
                +l*(l+1)*left[m*value_stride]*mixed*right[(width+m)*value_stride]
                +l*(l+1)*left[(width+m)*value_stride]*mixed*right[m*value_stride])/(width*width);
            ion+=term;
        }
        v[0].real-=ion;v[3].real-=ion;
    }
    bool valid=isfinite(ion);
    for(int k=0;k<4;k++)valid=valid&&isfinite(v[k].real)&&isfinite(v[k].imag);
    if constexpr(Gradients) for(int k=0;k<12;k++)valid=valid&&isfinite(f[k].real)&&isfinite(f[k].imag);
    if(!valid){return 2;}
    out[0]=ion;
    for(int k=0;k<4;k++){out[1+k]=v[k].real;out[5+k]=v[k].imag;}
    for(int k=0;k<12;k++){out[9+k]=f[k].real;out[21+k]=f[k].imag;}
    return 0;
}
// Preserve the existing force entry point and its arithmetic exactly.
__device__ inline int evaluate(int l,int split,const double* epsilon,const double* left,const double* right,
        const double* gradient,int value_stride,int gradient_stride,double* out) {
    return evaluate_impl<true>(l,split,epsilon,left,right,gradient,value_stride,gradient_stride,out);
}
// H0 only consumes values; gradient storage may be absent or poisoned.
__device__ inline int matrix(int l,int split,const double* epsilon,const double* left,const double* right,
        int value_stride,double* out) {
    return evaluate_impl<false>(l,split,epsilon,left,right,nullptr,value_stride,0,out);
}
}
