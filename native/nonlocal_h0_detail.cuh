// SPDX-License-Identifier: GPL-3.0-or-later
#pragma once
#include "nonlocal_h0.hpp"
#include "soc_projector_detail.cuh"
#include <cmath>
namespace remag::h0_detail {
__device__ bool span(int start,int count,int limit) {
    return start>=0&&start<=limit&&count>=0&&count<=limit-start;
}
__device__ int center_error(const NonlocalH0View& v,int i) {
    const auto c=v.center[i];
    if(!span(c.neighbor,c.neighbors,v.neighbors)||!span(c.block,c.blocks,v.blocks)||
       c.projectors<1||c.query<0||size_t(c.query)+size_t(c.neighbors)*c.projectors>size_t(v.queries))return 1;
    for(int b=c.block;b<c.block+c.blocks;b++) {
        const auto p=v.block[b];
        if(p.l<0||p.l>5||(!v.soc&&p.l)||!span(p.offset,p.l?2*(2*p.l+1):1,c.projectors))return 1;
    }
    // Disjoint center slices preserve the upstream center traversal order.
    if(i) {
        const auto previous=v.center[i-1];
        if(!span(previous.neighbor,previous.neighbors,v.neighbors)||
           c.neighbor<previous.neighbor+previous.neighbors)return 1;
    }
    return 0;
}
__device__ int row_error(const NonlocalH0View& v,int row) {
    const int begin=v.ptr[row],count=v.num[row];
    if(!span(begin,count,v.capacity))return 1;
    // CSR rows may occupy disjoint storage spans in any order. Empty rows
    // own no entries, regardless of their offset. Validate before any writer
    // runs, so independent entry threads never race on a physical element.
    for(int other=0;other<row;other++) {
        const int previous=v.ptr[other],n=v.num[other];
        if(!span(previous,n,v.capacity))return 1;
        if(count&&n&&begin<previous+n&&previous<begin+count)return 1;
    }
    const int first=v.row_ptr[row],end=v.row_ptr[row+1];
    if(first<0||first>end||end>v.local_neighbors||(!row&&first)||
       (row==v.rows-1&&end!=v.local_neighbors))return 1;
    for(int k=first;k<end;k++) {
        int i=v.row_neighbors[k];
        if(i<0||i>=v.neighbors||v.neighbor[i].row!=row+1)return 1;
        if(k>first&&v.row_neighbors[k-1]>=i)return 1;
    }
    for(int i=begin;i<begin+count;i++)if(v.list[i]<1||v.list[i]>v.columns)return 1;
    return 0;
}
__device__ int neighbor_error(const NonlocalH0View& v,int i) {
    const auto n=v.neighbor[i];
    if(n.row<0||n.row>v.rows||n.column<1||n.column>v.columns||n.center<0||n.center>=v.centers)return 1;
    const auto c=v.center[n.center];
    if(!span(c.neighbor,c.neighbors,v.neighbors)||i<c.neighbor||i>=c.neighbor+c.neighbors)return 1;
    if(n.row) {
        const int start=v.row_ptr[n.row-1],end=v.row_ptr[n.row];
        if(start<0||start>end||end>v.local_neighbors)return 1;
        // Independent coverage check prevents a truncated row list from
        // silently dropping a projector contribution.
        int lo=start,hi=end;
        while(lo<hi){int mid=lo+(hi-lo)/2;if(v.row_neighbors[mid]<i)lo=mid+1;else hi=mid;}
        if(lo==end||v.row_neighbors[lo]!=i)return 1;
    }
    return 0;
}
// Called only after the complete topology validation kernel has succeeded.
__device__ int add_entry(const NonlocalH0View& v,int row,int index,double* h,H0Complex* z) {
    const int column=v.list[index];
    bool first=true;
    for(int j=v.ptr[row];j<index;j++)if(v.list[j]==column){first=false;break;}
    if(!first&&!v.soc)return 0;
    double value=first?h[index]:0.;H0Complex spin[4]={};
    if(v.soc)for(int k=0;k<4;k++)spin[k]=z[size_t(k)*v.capacity+index];
    for(int position=v.row_ptr[row];position<v.row_ptr[row+1];position++) {
        const int i=v.row_neighbors[position];
        const auto c=v.center[v.neighbor[i].center];
        const size_t left=size_t(c.query)+size_t(i-c.neighbor)*c.projectors;
        double scalar=0.;H0Complex op[4]={};
        for(int j=c.neighbor;j<c.neighbor+c.neighbors;j++) {
            if(v.neighbor[j].column!=column)continue;
            const size_t right=size_t(c.query)+size_t(j-c.neighbor)*c.projectors;
            for(int b=c.block;b<c.block+c.blocks;b++) {
                const auto block=v.block[b];
                const double* a=v.integrals+4*(left+block.offset);
                const double* d=v.integrals+4*(right+block.offset);
                if(!block.l) {
                    if(!isfinite(block.epsilon[0])||!isfinite(a[0])||!isfinite(d[0]))return 2;
                    const double term=block.epsilon[0]*a[0]*d[0];
                    if(!v.soc||v.split)scalar+=term;
                    else {op[0].real+=term;op[1].real+=term;}
                } else {
                    double result[33];
                    const int error=soc_detail::matrix(block.l,v.split,block.epsilon,a,d,4,result);
                    if(error)return error;
                    scalar+=result[0];
                    const int order[4]={0,3,2,1};
                    for(int k=0;k<4;k++){op[k].real+=result[1+order[k]];op[k].imag+=result[5+order[k]];}
                }
            }
        }
        if(first){value+=scalar;if(!isfinite(value))return 2;}
        if(v.soc)for(int k=0;k<4;k++) {
            spin[k].real+=op[k].real;spin[k].imag+=op[k].imag;
            if(!isfinite(spin[k].real)||!isfinite(spin[k].imag))return 2;
        }
    }
    if(first)h[index]=value;
    if(v.soc)for(int k=0;k<4;k++)z[size_t(k)*v.capacity+index]=spin[k];
    return 0;
}
}
