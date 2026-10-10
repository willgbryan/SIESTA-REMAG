// SPDX-License-Identifier: GPL-3.0-or-later
// Included in density_matrix.cu's anonymous namespace. Contract each orbital
// pair once; image phases are applied after the occupation-weighted state sum.
template<int Mode> __global__ void contract_unique(int pairs,int nev,
        const int* rows,const int* cols,const void* values,const double* occ,
        const double* energies,double cutoff,double* scratch) {
    const int lane=threadIdx.x%32,pair=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(pair>=pairs)return;
    const int i=rows[pair],j=cols[pair];
    constexpr int components=Mode==0?1:(Mode==1?2:8);
    double ds[components]={},es[components]={};
    for(int state=lane;state<nev;state+=32) {
        const double q=occ[state];
        if(fabs(q)<=cutoff)continue;
        double v[components]={};
        if constexpr(Mode==0) {
            const auto z=static_cast<const double*>(values);
            v[0]=z[size_t(i)*nev+state]*z[size_t(j)*nev+state];
        } else if constexpr(Mode==1) {
            const auto z=static_cast<const Z*>(values);
            const Z a=product(z[size_t(i)*nev+state],z[size_t(j)*nev+state]);
            v[0]=a.x;v[1]=a.y;
        } else {
            const auto z=static_cast<const Z*>(values);
            const Z iu=z[size_t(2*i)*nev+state],id=z[size_t(2*i+1)*nev+state];
            const Z ju=z[size_t(2*j)*nev+state],jd=z[size_t(2*j+1)*nev+state];
            const Z a=product(iu,ju),b=product(id,jd),c=product(iu,jd),f=product(id,ju);
            v[0]=a.x;v[1]=a.y;v[2]=b.x;v[3]=b.y;
            v[4]=c.x;v[5]=c.y;v[6]=f.x;v[7]=f.y;
        }
        for(int c=0;c<components;c++) {
            ds[c]+=q*v[c];es[c]+=(q*energies[state])*v[c];
        }
    }
    for(int c=0;c<components;c++) {
        const double d=reduce(ds[c]),e=reduce(es[c]);
        if(!lane) {
            scratch[size_t(c)*pairs+pair]=d;
            scratch[size_t(c+components)*pairs+pair]=e;
        }
    }
}
template<int Mode> __global__ void scatter_unique(int pairs,int unique,int nnz,int nh,int outspin,
        const int* indices,const int* mapping,const double* xyz,const double* scratch,
        double kx,double ky,double kz,double* d,double* e) {
    const int pair=blockIdx.x*blockDim.x+threadIdx.x;
    if(pair>=pairs)return;
    const int group=mapping[pair],ind=indices[pair];
    constexpr int components=Mode==0?1:(Mode==1?2:8);
    double co=1.,si=0.;
    if constexpr(Mode!=0) {
        const double angle=kx*xyz[3*pair]+ky*xyz[3*pair+1]+kz*xyz[3*pair+2];
        co=cos(angle);si=sin(angle);
    }
    for(int energy=0;energy<2;energy++) {
        const double* bank=scratch+size_t(energy*components)*unique;
        double* output=energy?e:d;
        if constexpr(Mode==0) output[size_t(outspin)*nnz+ind]+=bank[group];
        else if constexpr(Mode==1) {
            const Z a=phase(make_cuDoubleComplex(bank[group],bank[unique+group]),co,si);
            output[size_t(outspin)*nnz+ind]+=a.x;
        } else {
            const Z a=phase(make_cuDoubleComplex(bank[group],bank[unique+group]),co,si);
            const Z b=phase(make_cuDoubleComplex(bank[2*unique+group],bank[3*unique+group]),co,si);
            Z c=phase(make_cuDoubleComplex(bank[4*unique+group],bank[5*unique+group]),co,si);
            const Z f=phase(make_cuDoubleComplex(bank[6*unique+group],bank[7*unique+group]),co,si);
            if(nh==4)c=make_cuDoubleComplex(.5*(c.x+f.x),.5*(c.y-f.y));
            const double values[8]={a.x,b.x,c.x,-c.y,a.y,b.y,f.x,f.y};
            const int count=energy?4:nh;
            for(int component=0;component<count;component++)output[size_t(component)*nnz+ind]+=values[component];
        }
    }
}
