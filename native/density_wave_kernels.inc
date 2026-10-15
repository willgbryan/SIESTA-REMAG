// SPDX-License-Identifier: GPL-3.0-or-later
// Included in density_matrix.cu's anonymous namespace, after the per-job
// kernels. A wave of sampled-k jobs (lanes) in one pass: each output element
// is loaded once, the jobs' contributions are added to it in job order with
// exactly the arithmetic of build / contract_unique / scatter_unique, and it
// is stored once. The sums are therefore those of the per-job sequence.
// Vectors, occupations, energies and stages are per lane, at fixed strides.
template<int Mode> __global__ void build_wave(int pairs,int nnz,int nh,int nev,int lanes,
        const int* rows,const int* cols,const int* indices,const double* xyz,const void* values,size_t stride,
        const double* occupations,const double* eigenvalues,double cutoff,const remag::DensityJobStage* stages,
        double* d,double* e) {
    static_assert(Mode==1||Mode==2,"sampled-k modes only");
    const int lane=threadIdx.x%32,pair=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(pair>=pairs) return;
    const int i=rows[pair],j=cols[pair],ind=indices[pair];
    const int nd=Mode==2?nh:1,ne=Mode==2?4:1,outs=Mode==1?nh:1;
    double accd[2][8]={},acce[2][4]={};
    if(lane==0) for(int o=0;o<outs;o++) {
        for(int s=0;s<nd;s++) accd[o][s]=d[(size_t)(s+o)*nnz+ind];
        for(int s=0;s<ne;s++) acce[o][s]=e[(size_t)(s+o)*nnz+ind];
    }
    for(int job=0;job<lanes;job++) {
        const remag::DensityJobStage st=stages[job];
        const double* occ=occupations+size_t(job)*nev;const double* energies=eigenvalues+size_t(job)*nev;
        const Z* z=static_cast<const Z*>(values)+size_t(job)*stride;
        double angle=st.k[0]*xyz[3*pair]+st.k[1]*xyz[3*pair+1]+st.k[2]*xyz[3*pair+2];
        double co=cos(angle),si=sin(angle),ds[8]={},es[4]={};
        for(int state=lane;state<nev;state+=32) {
            double q=occ[state];
            if(fabs(q)<=cutoff) continue;
            double v[8]={};
            if constexpr(Mode==1) {
                v[0]=phase(product(z[(size_t)i*nev+state],z[(size_t)j*nev+state]),co,si).x;
            } else {
                Z iu=z[(size_t)(2*i)*nev+state],id=z[(size_t)(2*i+1)*nev+state];
                Z ju=z[(size_t)(2*j)*nev+state],jd=z[(size_t)(2*j+1)*nev+state];
                Z a=phase(product(iu,ju),co,si),b=phase(product(id,jd),co,si);
                Z c=phase(product(iu,jd),co,si),f=phase(product(id,ju),co,si);
                if(nh==4) c=make_cuDoubleComplex(.5*(c.x+f.x),.5*(c.y-f.y));
                v[0]=a.x;v[1]=b.x;v[2]=c.x;v[3]=-c.y;
                v[4]=a.y;v[5]=b.y;v[6]=f.x;v[7]=f.y;
            }
            constexpr int cd=Mode==2?8:1,ce=Mode==2?4:1;
            for(int s=0;s<cd;s++) ds[s]+=q*v[s];
            for(int s=0;s<ce;s++) es[s]+=(q*energies[state])*v[s];
        }
        const int o=Mode==1?st.outspin:0;
        for(int s=0;s<nd;s++) { double v=reduce(ds[s]);if(lane==0) accd[o][s]+=v; }
        for(int s=0;s<ne;s++) { double v=reduce(es[s]);if(lane==0) acce[o][s]+=v; }
    }
    if(lane==0) for(int o=0;o<outs;o++) {
        for(int s=0;s<nd;s++) d[(size_t)(s+o)*nnz+ind]=accd[o][s];
        for(int s=0;s<ne;s++) e[(size_t)(s+o)*nnz+ind]=acce[o][s];
    }
}
// contract_unique for lane blockIdx.y, into that lane's slice of the scratch.
template<int Mode> __global__ void contract_unique_wave(int pairs,int nev,const int* rows,const int* cols,const void* values,
        size_t stride,const double* occupations,const double* eigenvalues,double cutoff,double* scratch,size_t scratch_stride) {
    const int job=blockIdx.y;
    const int lane=threadIdx.x%32,pair=(blockIdx.x*blockDim.x+threadIdx.x)/32;
    if(pair>=pairs)return;
    const double* occ=occupations+size_t(job)*nev;const double* energies=eigenvalues+size_t(job)*nev;
    const Z* z=static_cast<const Z*>(values)+size_t(job)*stride;double* out=scratch+size_t(job)*scratch_stride;
    const int i=rows[pair],j=cols[pair];
    constexpr int components=Mode==1?2:8;
    double ds[components]={},es[components]={};
    for(int state=lane;state<nev;state+=32) {
        const double q=occ[state];
        if(fabs(q)<=cutoff)continue;
        double v[components]={};
        if constexpr(Mode==1) {
            const Z a=product(z[size_t(i)*nev+state],z[size_t(j)*nev+state]);
            v[0]=a.x;v[1]=a.y;
        } else {
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
        const double dsum=reduce(ds[c]),esum=reduce(es[c]);
        if(!lane) {
            out[size_t(c)*pairs+pair]=dsum;
            out[size_t(c+components)*pairs+pair]=esum;
        }
    }
}
// scatter_unique over all lanes, in job order, for one sparse pair per thread.
template<int Mode> __global__ void scatter_unique_wave(int pairs,int unique,int nnz,int nh,int lanes,
        const int* indices,const int* mapping,const double* xyz,const double* scratch,size_t scratch_stride,
        const remag::DensityJobStage* stages,double* d,double* e) {
    const int pair=blockIdx.x*blockDim.x+threadIdx.x;
    if(pair>=pairs)return;
    const int group=mapping[pair],ind=indices[pair];
    constexpr int components=Mode==1?2:8;
    const int nd=Mode==2?nh:1,ne=Mode==2?4:1,outs=Mode==1?nh:1;
    double accd[2][8]={},acce[2][4]={};
    for(int o=0;o<outs;o++) {
        for(int s=0;s<nd;s++) accd[o][s]=d[size_t(s+o)*nnz+ind];
        for(int s=0;s<ne;s++) acce[o][s]=e[size_t(s+o)*nnz+ind];
    }
    for(int job=0;job<lanes;job++) {
        const remag::DensityJobStage st=stages[job];
        const double angle=st.k[0]*xyz[3*pair]+st.k[1]*xyz[3*pair+1]+st.k[2]*xyz[3*pair+2];
        const double co=cos(angle),si=sin(angle);
        const int o=Mode==1?st.outspin:0;
        for(int energy=0;energy<2;energy++) {
            const double* bank=scratch+size_t(job)*scratch_stride+size_t(energy*components)*unique;
            double* acc=energy?acce[o]:accd[o];
            if constexpr(Mode==1) {
                const Z a=phase(make_cuDoubleComplex(bank[group],bank[unique+group]),co,si);
                acc[0]+=a.x;
            } else {
                const Z a=phase(make_cuDoubleComplex(bank[group],bank[unique+group]),co,si);
                const Z b=phase(make_cuDoubleComplex(bank[2*unique+group],bank[3*unique+group]),co,si);
                Z c=phase(make_cuDoubleComplex(bank[4*unique+group],bank[5*unique+group]),co,si);
                const Z f=phase(make_cuDoubleComplex(bank[6*unique+group],bank[7*unique+group]),co,si);
                if(nh==4)c=make_cuDoubleComplex(.5*(c.x+f.x),.5*(c.y-f.y));
                const double values[8]={a.x,b.x,c.x,-c.y,a.y,b.y,f.x,f.y};
                const int count=energy?4:nh;
                for(int component=0;component<count;component++)acc[component]+=values[component];
            }
        }
    }
    for(int o=0;o<outs;o++) {
        for(int s=0;s<nd;s++) d[size_t(s+o)*nnz+ind]=accd[o][s];
        for(int s=0;s<ne;s++) e[size_t(s+o)*nnz+ind]=acce[o][s];
    }
}
