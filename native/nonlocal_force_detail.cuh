// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
namespace remag::nonlocal_detail {
struct Complex {double real,imag;};
__device__ Complex multiply(Complex a,Complex b){return {a.real*b.real-a.imag*b.imag,a.real*b.imag+a.imag*b.real};}
__device__ void evaluate(int kind,int split,double diagonal,const double* d,const double* s,const double* op,double* de,double* df) {
        if(kind==0) {
            const double coefficient=diagonal*s[0];
            de[0]=coefficient*s[1]*s[2];
            for(int axis=0;axis<3;axis++)df[axis]=2.*coefficient*s[2]*s[3+axis];
        } else if(kind==1) {
            const double potential=s[0]*s[1]*s[2];
            if(split)de[0]=diagonal*potential;
            else {de[1]=potential*d[0];de[2]=potential*d[1];de[5]=potential*d[4];de[6]=potential*d[5];}
            const double coefficient=2.*diagonal*(s[0]*s[2]);
            for(int axis=0;axis<3;axis++)df[axis]=coefficient*s[3+axis];
        } else {
            const Complex weights[4]={{d[0],d[4]},{d[1],d[5]},{d[6],-d[7]},{d[2],d[3]}};
            const int order[4]={0,3,2,1}; // 11,22,12,21 in Fortran storage
            de[0]=diagonal*op[0];
            for(int k=0;k<4;k++) {
                const int index=order[k];
                const auto value=multiply({op[1+index],op[5+index]},weights[k]);
                de[1+k]=value.real;de[5+k]=value.imag;
            }
            for(int axis=0;axis<3;axis++) {
                Complex sum={0.,0.};
                for(int k=0;k<4;k++) {
                    const int index=axis+3*order[k];
                    const auto term=multiply(weights[k],{op[9+index],op[21+index]});
                    sum.real+=term.real;sum.imag+=term.imag;
                }
                df[axis]=2.*sum.real;
            }
        }
}
}
