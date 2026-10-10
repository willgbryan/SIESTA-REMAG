// SPDX-License-Identifier: GPL-3.0-or-later
// Direct port of pinned atmfuncs:all_phi, radial:rad_get,
// interpolation:evaluate_spline_dx/interpolate_interval and spher_harm:rlylm.
#include "orbital_basis.hpp"
#include <cmath>
#include <climits>
namespace remag {
namespace {
__device__ void fail(int* status,int code) {atomicMax(status,code);}
__device__ double normalization(int l,int m,double pi) {
    m=m<0?-m:m;double fac=(2*l+1)/(4*pi);
    for(int i=l-m+1;i<=l+m;i++) fac=fac/i;
    double c=sqrt(fac);if(m) c=c*sqrt(2.);return c;
}
__device__ void harmonic(int lmax,int l,int signed_m,const double* r,double pi,double& value,double* grad) {
    const int m=signed_m<0?-signed_m:signed_m,ilm=l*l+l+signed_m;
    const double c=normalization(l,m,pi);value=0.;grad[0]=grad[1]=grad[2]=0.;
    if(lmax<=2) {
        switch(ilm) {
        case 0:value=c;break;
        case 1:value=-(c*r[1]);grad[1]=-c;break;
        case 2:value=c*r[2];grad[2]=c;break;
        case 3:value=-(c*r[0]);grad[0]=-c;break;
        case 4:value=c*6*r[0]*r[1];grad[0]=c*6*r[1];grad[1]=c*6*r[0];break;
        case 5:value=(-c)*3*r[1]*r[2];grad[1]=(-c)*3*r[2];grad[2]=(-c)*3*r[1];break;
        case 6:value=c*.5*(2*r[2]*r[2]-r[0]*r[0]-r[1]*r[1]);grad[0]=(-c)*r[0];grad[1]=(-c)*r[1];grad[2]=c*2*r[2];break;
        case 7:value=(-c)*3*r[0]*r[2];grad[0]=(-c)*3*r[2];grad[2]=(-c)*3*r[0];break;
        case 8:value=c*3*(r[0]*r[0]-r[1]*r[1]);grad[0]=c*6*r[0];grad[1]=(-c)*6*r[1];break;
        }
        return;
    }
    const double size=sqrt(r[0]*r[0]+r[1]*r[1]+r[2]*r[2]);
    if(size<1.e-14) {
        if(ilm==0)value=c;
        if(ilm==1)grad[1]=-c;
        if(ilm==2)grad[2]=c;
        if(ilm==3)grad[0]=-c;
        return;
    }
    double rx=r[0]/size,ry=r[1]/size,rz=r[2]/size,rxy=sqrt(rx*rx+ry*ry);
    if(rxy<1.e-14) {rx=1.e-14;rxy=sqrt(rx*rx+ry*ry);}
    double p[7][7]{},zp[7][7]{};
    for(int order=lmax;order>=0;order--) {
        p[order][order+1]=0.;p[order][order]=1.;double fac=1.;
        for(int i=1;i<=order;i++) {p[order][order]=-(p[order][order]*fac*rxy);fac=fac+2.;}
        p[order+1][order]=rz*(2*order+1)*p[order][order];
        for(int ll=order+2;ll<=lmax;ll++)
            p[ll][order]=(rz*(2*ll-1)*p[ll-1][order]-(ll+order-1)*p[ll-2][order])/(ll-order);
        for(int ll=order;ll<=lmax;ll++) zp[ll][order]=-((order*p[ll][order]*rz/rxy+p[ll][order+1])/rxy);
    }
    double rl[7]{};rl[0]=0.;rl[1]=1.;for(int ll=1;ll<=lmax;ll++)rl[ll+1]=rl[ll]*size;
    const double cosphi=rx/rxy,sinphi=ry/rxy;double cosm=1.,sinm=0.;
    for(int mm=0;mm<m;mm++) {double oldc=cosm,olds=sinm;cosm=oldc*cosphi-olds*sinphi;sinm=oldc*sinphi+olds*cosphi;}
    double gy[3],yy;
    if(signed_m<0) {
        gy[0]=(-zp[l][m])*rx*rz*sinm-p[l][m]*m*cosm*sinphi/rxy;
        gy[1]=(-zp[l][m])*ry*rz*sinm+p[l][m]*m*cosm*cosphi/rxy;
        gy[2]=zp[l][m]*rxy*rxy*sinm;yy=c*p[l][m]*sinm;
    } else {
        gy[0]=(-zp[l][m])*rx*rz*cosm+p[l][m]*m*sinm*sinphi/rxy;
        gy[1]=(-zp[l][m])*ry*rz*cosm-p[l][m]*m*sinm*cosphi/rxy;
        gy[2]=zp[l][m]*rxy*rxy*cosm;yy=c*p[l][m]*cosm;
    }
    value=rl[l+1]*yy;
    grad[0]=rx*l*rl[l]*yy+rl[l+1]*gy[0]*c/size;
    grad[1]=ry*l*rl[l]*yy+rl[l+1]*gy[1]*c/size;
    grad[2]=rz*l*rl[l]*yy+rl[l+1]*gy[2]*c/size;
}
__device__ bool descriptor(const OrbitalBasisView& b,int index,int* status) {
    const int* d=b.orbitals+4*index;const double* v=b.radial+2*index;
    if(d[0]<0||d[0]==1||d[1]<0||d[1]>b.sample_count||d[0]>b.sample_count-d[1]||d[2]<0||d[2]>5||d[3]<-d[2]||d[3]>d[2]||
       !isfinite(v[0])||v[0]<0.||!isfinite(v[1])||v[1]<=0.) {fail(status,1);return false;}return true;
}
__device__ void orbital(const OrbitalBasisView& b,int species,int local,const double* r,double* result,int* status) {
    for(int i=0;i<4;i++)result[i]=0.;
    if(species<0||species>=b.species_count||local<0) {fail(status,1);return;}
    const int start=b.species[2*species],count=b.species[2*species+1];
    if(start<0||count<1||start>b.orbital_count||count>b.orbital_count-start||local>=count) {fail(status,1);return;}
    if(!isfinite(r[0])||!isfinite(r[1])||!isfinite(r[2])) {fail(status,2);return;}
    const double r2=r[0]*r[0]+r[1]*r[1]+r[2]*r[2],radius=sqrt(r2)+1.e-20;
    if(!isfinite(radius)) {fail(status,2);return;}
    double cutoff2=0.;int maxlm=0;
    for(int i=start;i<start+count;i++) {
        if(!descriptor(b,i,status))return;
        const double cutoff=b.radial[2*i];cutoff2=fmax(cutoff2,cutoff*cutoff);
        if(radius<cutoff) {const int l=b.orbitals[4*i+2],m=b.orbitals[4*i+3];maxlm=max(maxlm,l*(l+1)+m+1);}
    }
    const int index=start+local;const int* d=b.orbitals+4*index;const double cutoff=b.radial[2*index],delta=b.radial[2*index+1];
    if(r2>=cutoff2||radius>=cutoff||!d[0])return;
    const int lmax=int(floor(sqrt(double(maxlm))+.5))-1;
    if(lmax<d[2]||lmax>5) {fail(status,1);return;}
    const double xmax=(d[0]-1)*delta,tol=delta*1.e-6;
    if(radius< -tol||radius>xmax+tol) {fail(status,2);return;}
    const int kl=min(d[0]-2,max(0,int(floor(radius/delta)))),kh=kl+1;
    const double xl=kl*delta,xh=kh*delta,dx=xh-xl;
    const double yl=b.values[d[1]+kl],yh=b.values[d[1]+kh],d2l=b.second[d[1]+kl],d2h=b.second[d[1]+kh];
    if(!isfinite(yl)||!isfinite(yh)||!isfinite(d2l)||!isfinite(d2h)) {fail(status,2);return;}
    const double a=(xh-radius)/dx,bb=(radius-xl)/dx;
    const double radial=a*yl+bb*yh+((a*a*a-a)*d2l+(bb*bb*bb-bb)*d2h)*(dx*dx)/6.;
    const double da=-1./dx,db=1./dx;
    const double derivative=da*yl+db*yh+(da*(3*(a*a)-1.)*d2l+db*(3*(bb*bb)-1.)*d2h)*(dx*dx)/6.;
    double angular,gradient[3];harmonic(lmax,d[2],d[3],r,b.pi,angular,gradient);
    result[0]=radial*angular;
    for(int i=0;i<3;i++) result[i+1]=derivative*angular*r[i]/radius+radial*gradient[i];
    for(int i=0;i<4;i++)if(!isfinite(result[i]))fail(status,2);
}
__global__ void queries(OrbitalBasisView b,int count,const int* species,const int* local,const double* xyz,double* result,int* status) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<count;i+=blockDim.x*gridDim.x) orbital(b,species[i],local[i],xyz+size_t(3)*i,result+size_t(4)*i,status);
}
__global__ void geometry(OrbitalBasisView b,OrbitalGeometryView g,double* phi,double* grad,double* dx,int* status) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(g.entries)*g.subpoints;i+=size_t(blockDim.x)*gridDim.x) {
        const int entry=i/g.subpoints,sub=i%g.subpoints,io=g.entry_orbital[entry]-1,offset=g.entry_offset[entry]-1;
        phi[i]=0.;for(int k=0;k<3;k++)grad[3*i+k]=dx[3*i+k]=0.;
        if(io<0||io>=g.orbitals||offset<0||offset>=g.offsets) {fail(status,1);continue;}
        const int atom=g.orbital_atom[io]-1;
        if(atom<0||atom>=g.atoms) {fail(status,1);continue;}
        double r[3],v[4];for(int k=0;k<3;k++)r[k]=g.offset[3*offset+k]+g.subpoint[3*sub+k]-g.atom_offset[3*atom+k];
        orbital(b,g.atom_species[atom]-1,g.orbital_local[io]-1,r,v,status);phi[i]=v[0];
        for(int k=0;k<3;k++) {grad[3*i+k]=v[k+1];dx[3*i+k]=r[k];}
    }
}
// Literal PhiOnMesh -> phiatm value path. Do not replace with orbital():
// all_phi chooses a species-wide harmonic order and a different cutoff.
__device__ float mesh_orbital(const OrbitalBasisView& b,int species,int local,const double* r,int* status) {
    if(species<0||species>=b.species_count||local<0) {fail(status,1);return 0.f;}
    const int start=b.species[2*species],count=b.species[2*species+1];
    if(start<0||count<1||start>b.orbital_count||count>b.orbital_count-start||local>=count) {fail(status,1);return 0.f;}
    const int index=start+local;
    if(!descriptor(b,index,status))return 0.f;
    if(!isfinite(r[0])||!isfinite(r[1])||!isfinite(r[2])) {fail(status,2);return 0.f;}
    const int* d=b.orbitals+4*index;
    const double cutoff=b.radial[2*index],delta=b.radial[2*index+1];
    const double r2=r[0]*r[0]+r[1]*r[1]+r[2]*r[2],radius=sqrt(r2)+1.e-20;
    if(!isfinite(radius)) {fail(status,2);return 0.f;}
    if(!(r2<cutoff*cutoff)||radius>cutoff-1.e-12||!d[0])return 0.f;
    const double xmax=(d[0]-1)*delta,tol=delta*1.e-6;
    if(radius< -tol||radius>xmax+tol) {fail(status,2);return 0.f;}
    const int kl=min(d[0]-2,max(0,int(floor(radius/delta)))),kh=kl+1;
    const double xl=kl*delta,xh=kh*delta,dx=xh-xl;
    const double yl=b.values[d[1]+kl],yh=b.values[d[1]+kh],d2l=b.second[d[1]+kl],d2h=b.second[d[1]+kh];
    if(!isfinite(yl)||!isfinite(yh)||!isfinite(d2l)||!isfinite(d2h)) {fail(status,2);return 0.f;}
    const double a=(xh-radius)/dx,bb=(radius-xl)/dx;
    const double radial=a*yl+bb*yh+((a*a*a-a)*d2l+(bb*bb*bb-bb)*d2h)*(dx*dx)/6.;
    double angular,gradient[3];harmonic(d[2],d[2],d[3],r,b.pi,angular,gradient);
    const float value=static_cast<float>(radial*angular);
    if(!isfinite(value)) {fail(status,2);return 0.f;}
    return value;
}
__global__ void mesh_values(OrbitalBasisView b,OrbitalGeometryView g,float* phi,int* status) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<size_t(g.entries)*g.subpoints;i+=size_t(blockDim.x)*gridDim.x) {
        const int entry=i/g.subpoints,sub=i%g.subpoints,io=g.entry_orbital[entry]-1,offset=g.entry_offset[entry]-1;
        phi[i]=0.f;
        if(io<0||io>=g.orbitals||offset<0||offset>=g.offsets) {fail(status,1);continue;}
        const int atom=g.orbital_atom[io]-1;
        if(atom<0||atom>=g.atoms) {fail(status,1);continue;}
        double r[3];for(int k=0;k<3;k++)r[k]=g.offset[3*offset+k]+g.subpoint[3*sub+k]-g.atom_offset[3*atom+k];
        phi[i]=mesh_orbital(b,g.atom_species[atom]-1,g.orbital_local[io]-1,r,status);
    }
}
bool valid(const OrbitalBasisView& b) {
    return b.species_count>0&&b.species_count<=10000&&b.orbital_count>0&&b.orbital_count<=1000000&&b.sample_count>=0&&
        std::isfinite(b.pi)&&b.pi>3.14&&b.pi<3.15&&b.species&&b.orbitals&&b.radial&&b.values&&b.second;
}
__global__ void compare_geometry(size_t n,const double* phi,const double* gradient,const double* displacement,
        const double* rp,const double* rg,const double* rd,OrbitalComparison* result,int* status) {
    unsigned long long compared=0,mismatches=0,nonfinite=0;
    double absolute[3]{},maximum_scaled[3]{};
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<7*n;i+=size_t(blockDim.x)*gridDim.x) {
        const int kind=i<n?0:i<4*n?1:2;
        const size_t k=kind==0?i:kind==1?i-n:i-4*n;
        const double a=(kind==0?phi:kind==1?gradient:displacement)[k];
        const double b=(kind==0?rp:kind==1?rg:rd)[k];
        compared++;
        const double error=fabs(a-b);
        if(!isfinite(a)||!isfinite(b)||!isfinite(error)) {nonfinite++;continue;}
        const double scaled=kind==2?error:error/(1.e-10+1.e-10*fabs(b));
        absolute[kind]=fmax(absolute[kind],error);maximum_scaled[kind]=fmax(maximum_scaled[kind],scaled);
        if(kind==2?error!=0.:scaled>1.) mismatches++;
    }
    if(compared) {
        atomicAdd(&result->compared,compared);
        if(mismatches){atomicAdd(&result->mismatches,mismatches);fail(status,3);}
        if(nonfinite){atomicAdd(&result->nonfinite,nonfinite);fail(status,2);}
        for(int kind=0;kind<3;kind++) {
            atomicMax(reinterpret_cast<unsigned long long*>(result->absolute+kind),__double_as_longlong(absolute[kind]));
            atomicMax(reinterpret_cast<unsigned long long*>(result->scaled+kind),__double_as_longlong(maximum_scaled[kind]));
        }
    }
}
}
cudaError_t evaluate_orbital_queries_device(const OrbitalBasisView& b,int count,const int* species,const int* local,const double* xyz,
        double* result,int* status,cudaStream_t stream) {
    if(!valid(b)||count<1||count>INT_MAX/4||!species||!local||!xyz||!result||!status||!stream)return cudaErrorInvalidValue;
    queries<<<(count+127)/128,128,0,stream>>>(b,count,species,local,xyz,result,status);return cudaGetLastError();
}
cudaError_t evaluate_orbital_geometry_device(const OrbitalBasisView& b,const OrbitalGeometryView& g,
        double* phi,double* gradient,double* displacement,int* status,cudaStream_t stream) {
    if(!valid(b)||g.entries<1||g.subpoints<1||g.subpoints>4096||size_t(g.entries)*g.subpoints>INT_MAX/4||
       g.orbitals<1||g.atoms<1||g.offsets<1||!g.entry_orbital||!g.entry_offset||!g.orbital_atom||!g.orbital_local||!g.atom_species||
       !g.offset||!g.subpoint||!g.atom_offset||!phi||!gradient||!displacement||!status||!stream)return cudaErrorInvalidValue;
    geometry<<<(size_t(g.entries)*g.subpoints+127)/128,128,0,stream>>>(b,g,phi,gradient,displacement,status);return cudaGetLastError();
}
cudaError_t evaluate_orbital_mesh_device(const OrbitalBasisView& b,const OrbitalGeometryView& g,
        float* phi,int* status,cudaStream_t stream) {
    if(!valid(b)||g.entries<0||g.subpoints<1||g.subpoints>4096||size_t(g.entries)*g.subpoints>INT_MAX/4||
       g.orbitals<1||g.atoms<1||g.offsets<1||!g.entry_orbital||!g.entry_offset||!g.orbital_atom||!g.orbital_local||!g.atom_species||
       !g.offset||!g.subpoint||!g.atom_offset||!phi||!status||!stream)return cudaErrorInvalidValue;
    if(!g.entries)return cudaSuccess;
    mesh_values<<<(size_t(g.entries)*g.subpoints+127)/128,128,0,stream>>>(b,g,phi,status);return cudaGetLastError();
}
cudaError_t compare_orbital_geometry_device(size_t points,const double* phi,const double* gradient,const double* displacement,
        const double* rp,const double* rg,const double* rd,OrbitalComparison* result,int* status,cudaStream_t stream) {
    if(points<1||points>size_t(INT_MAX)/7||!phi||!gradient||!displacement||!rp||!rg||!rd||!result||!status||!stream)return cudaErrorInvalidValue;
    const int blocks=int((7*points+255)/256);
    compare_geometry<<<blocks>1024?1024:blocks,256,0,stream>>>(points,phi,gradient,displacement,rp,rg,rd,result,status);
    return cudaGetLastError();
}
}
