// SPDX-License-Identifier: GPL-3.0-or-later
// Mesh functional from pinned GridXC cellxc.F90; see GRIDXC_NOTICE.
#include "pbe.hpp"
#include "pbe_math.hpp"
#include <algorithm>
#include <climits>
#include <cmath>
#include <vector>
#include <stdexcept>

namespace remag {
namespace {
constexpr int threads=256,fields=13;
void check(cudaError_t e) {if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e));}
__device__ bool shape_ok(const PbeProblem& p,int size,int spins) {
    return p.spins==spins&&p.mesh[0]>0&&p.mesh[1]>0&&p.mesh[2]>0&&
           static_cast<long long>(p.mesh[0])*p.mesh[1]<=INT_MAX/p.mesh[2]&&
           p.mesh[0]*p.mesh[1]*p.mesh[2]==size;
}
__global__ void initialize(const PbeProblem* p,PbeResult* out,int n1,int n2,int n3,int spins) {
    *out=PbeResult{};
    if(p->mesh[0]!=n1||p->mesh[1]!=n2||p->mesh[2]!=n3||p->spins!=spins) out->status=1;
    if(!isfinite(p->volume)||p->volume<=0.) out->status=2;
    for(int i=0;i<9;i++) if(!isfinite(p->gradient[i])) out->status=2;
    const auto b=p->gradient;
    double determinant=b[0]*(b[4]*b[8]-b[5]*b[7])-b[3]*(b[1]*b[8]-b[2]*b[7])+b[6]*(b[1]*b[5]-b[2]*b[4]);
    if(!isfinite(determinant)||determinant==0.) out->status=2;
}
__device__ double weight(int offset) {
    // Original nn=3 Lagrange derivative stencil, in increasing neighbor order.
    const double w[7]={-1./60.,3./20.,-3./4.,0.,3./4.,-3./20.,1./60.};
    return w[offset+3];
}
__device__ int neighbor(int index,const PbeProblem& p,int axis,int delta) {
    const int stride=axis==0?1:axis==1?p.mesh[0]:p.mesh[0]*p.mesh[1];
    const int coordinate=(index/stride)%p.mesh[axis];
    int shifted=(coordinate+delta)%p.mesh[axis];if(shifted<0) shifted+=p.mesh[axis];
    return index+(shifted-coordinate)*stride;
}
__device__ double finite_density(const double* density,int index,PbeResult* out) {
    double value=density[index];
    if(!isfinite(value)) {atomicMax(&out->status,2);return 0.;}
    return value;
}
__device__ void gradient(int index,int size,int spins,const PbeProblem& p,const double* rho,
                         double g[4][3],PbeResult* out) {
    for(int axis=0;axis<3;axis++) for(int offset=-3;offset<=3;offset++) {
        const int j=neighbor(index,p,axis,offset);
        for(int s=0;s<spins;s++) {
            const double value=finite_density(rho,j+size*s,out);
            for(int k=0;k<3;k++) g[s][k]+=p.gradient[k+3*axis]*weight(offset)*value;
        }
    }
}
__global__ void evaluate(int size,int spins,const PbeProblem* problem,const double* density,
                         PbePoint* points,PbeResult* out) {
    const auto p=*problem;const bool valid=shape_ok(p,size,spins);
    for(int index=blockIdx.x*blockDim.x+threadIdx.x;index<size;index+=gridDim.x*blockDim.x) {
        PbePoint point{};
        if(valid) {
            double d[4]{},g[4][3]{};
            for(int s=0;s<spins;s++) d[s]=finite_density(density,index+size*s,out);
            double total=d[0]+(spins==1?0.:d[1]);
            if(!isfinite(total)) atomicMax(&out->status,2);
            if(total>=1.e-15) {
                gradient(index,size,spins,p,density,g,out);
                point=pbe_point(spins,d,g);
                bool finite=isfinite(point.x)&&isfinite(point.c);
                for(int s=0;s<spins;s++) {
                    finite=finite&&isfinite(point.dx[s])&&isfinite(point.dc[s]);
                    for(int k=0;k<3;k++) finite=finite&&isfinite(point.gx[s][k])&&isfinite(point.gc[s][k]);
                }
                if(!finite) {atomicMax(&out->status,2);point=PbePoint{};}
            }
        }
        points[index]=point;
    }
}
__global__ void assemble(int size,int spins,const PbeProblem* problem,const double* density,
                         const PbePoint* points,double* potential,double* partial,PbeResult* out) {
    __shared__ double sums[fields][threads];
    const auto p=*problem;const bool valid=shape_ok(p,size,spins);
    double values[fields]{};
    for(int index=blockIdx.x*blockDim.x+threadIdx.x;index<size;index+=gridDim.x*blockDim.x) {
        if(!valid) {for(int s=0;s<spins;s++) potential[index+size*s]=0.;continue;}
        const auto point=points[index];double d[4]{},g[4][3]{};
        for(int s=0;s<spins;s++) d[s]=finite_density(density,index+size*s,out);
        const double total=d[0]+(spins==1?0.:d[1]);
        const double ex=total*point.x,ec=total*point.c;
        values[0]+=ex;values[1]+=ec;values[2]+=ex;values[3]+=ec;
        for(int s=0;s<spins;s++) {
            double vx=point.dx[s],vc=point.dc[s];
            // Gather the adjoint contributions. Duplicate periodic neighbors
            // in small meshes retain their separate stencil coefficients.
            for(int axis=0;axis<3;axis++) for(int offset=-3;offset<=3;offset++) {
                const auto& source=points[neighbor(index,p,axis,-offset)];
                double x=0.,c=0.;
                for(int k=0;k<3;k++) {
                    x+=source.gx[s][k]*(p.gradient[k+3*axis]*weight(offset));
                    c+=source.gc[s][k]*(p.gradient[k+3*axis]*weight(offset));
                }
                vx+=x;vc+=c;
            }
            potential[index+size*s]=2*(vx+vc);
            if(!isfinite(potential[index+size*s])) atomicMax(&out->status,2);
            values[2]-=d[s]*vx;values[3]-=d[s]*vc;
        }
        // Density at a skipped point can still receive neighboring adjoint
        // potential, but its own pointwise gradient-stress contribution is zero.
        if(total>=1.e-15) {
            gradient(index,size,spins,p,density,g,out);
            for(int col=0;col<3;col++) for(int row=0;row<3;row++)
                for(int s=0;s<spins;s++) values[4+row+3*col]-=g[s][row]*(point.gx[s][col]+point.gc[s][col]);
        }
    }
    for(int field=0;field<fields;field++) sums[field][threadIdx.x]=values[field];
    __syncthreads();
    for(int step=threads/2;step;step/=2) {
        if(threadIdx.x<step) for(int field=0;field<fields;field++) sums[field][threadIdx.x]+=sums[field][threadIdx.x+step];
        __syncthreads();
    }
    if(threadIdx.x==0) for(int field=0;field<fields;field++) partial[fields*blockIdx.x+field]=sums[field][0];
}
// Local frame. The decision record lives for the process.
struct Frame {int decided,signed_axis;double axis[3];};
__global__ void frame_problem(const PbeProblem* outer,PbeProblem* inner) {
    *inner=*outer;inner->spins=outer->spins==4?2:0; // a wrong outer spin count fails shape checks
}
// Per-block sums of m and m m^T (12 values) while undecided.
__global__ void frame_moments(int size,const double* d,const Frame* frame,double* partial) {
    __shared__ double sums[12][threads];double value[12]{};
    if(!frame->decided) for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=gridDim.x*blockDim.x) {
        const double m[3]={2.*d[i+2*size],-2.*d[i+3*size],d[i]-d[i+size]};
        if(!isfinite(m[0])||!isfinite(m[1])||!isfinite(m[2])) continue;
        for(int a=0;a<3;a++) {value[9+a]+=m[a];for(int b=0;b<3;b++) value[a+3*b]+=m[a]*m[b];}
    }
    for(int f=0;f<12;f++) sums[f][threadIdx.x]=value[f];
    __syncthreads();
    for(int step=threads/2;step;step/=2) {
        if(threadIdx.x<step) for(int f=0;f<12;f++) sums[f][threadIdx.x]+=sums[f][threadIdx.x+step];
        __syncthreads();
    }
    if(threadIdx.x==0) for(int f=0;f<12;f++) partial[12*blockIdx.x+f]=sums[f][0];
}
// The host's rule: collinear up to sign if the largest eigenvalue of sum m m^T
// (power iteration from sum m) carries 99% of the trace. No magnetization yet:
// stay undecided, so a zero warm-up density cannot fix the choice.
__global__ void frame_decide(int blocks,const double* partial,Frame* frame) {
    if(frame->decided) return;
    double t[9]{},u[3]{};
    for(int i=0;i<blocks;i++) {for(int f=0;f<9;f++) t[f]+=partial[12*i+f];for(int a=0;a<3;a++) u[a]+=partial[12*i+9+a];}
    const double trace=t[0]+t[4]+t[8];
    if(!(trace>0.)||!isfinite(trace)) return;
    double v[3]={u[0],u[1],u[2]};
    if(norm3d(v[0],v[1],v[2])==0.) {v[0]=t[0];v[1]=t[4];v[2]=t[8];}
    if(norm3d(v[0],v[1],v[2])==0.) {v[0]=0.;v[1]=0.;v[2]=1.;}
    for(int it=0;it<200;it++) {
        double w[3];for(int a=0;a<3;a++) w[a]=t[a]*v[0]+t[a+3]*v[1]+t[a+6]*v[2];
        const double n=norm3d(w[0],w[1],w[2]);if(n==0.) break;
        for(int a=0;a<3;a++) v[a]=w[a]/n;
    }
    double quotient=0.;for(int a=0;a<3;a++) quotient+=v[a]*(t[a]*v[0]+t[a+3]*v[1]+t[a+6]*v[2]);
    frame->signed_axis=quotient>.99*trace;
    for(int a=0;a<3;a++) frame->axis[a]=frame->signed_axis?v[a]:(a==2?1.:0.);
    frame->decided=1;
}
__global__ void to_frame(int size,const double* d,const Frame* frame,double* channels,double* e,PbeResult* out) {
    const bool sign=frame->decided&&frame->signed_axis;const double* u=frame->axis;
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=gridDim.x*blockDim.x) {
        const double m[3]={2.*d[i+2*size],-2.*d[i+3*size],d[i]-d[i+size]},n=d[i]+d[i+size];
        const double a=norm3d(m[0],m[1],m[2]);double x[3]={0.,0.,0.};
        if(a>1.e-14) {
            const double s=sign?copysign(1.,m[0]*u[0]+m[1]*u[1]+m[2]*u[2]):1.;
            for(int k=0;k<3;k++) x[k]=s*m[k]/a;
        } else if(sign) for(int k=0;k<3;k++) x[k]=u[k];
        const double along=x[0]*m[0]+x[1]*m[1]+x[2]*m[2];
        channels[i]=.5*(n+along);channels[i+size]=.5*(n-along);
        for(int k=0;k<3;k++) e[i+size*k]=x[k];
        if(!isfinite(channels[i])||!isfinite(channels[i+size])) atomicMax(&out->status,2);
    }
}
// dE/dD11 = (v_up+v_dn)/2 + (v_up-v_dn)/2 e_z; SIESTA's off-diagonal
// components carry half the derivative: (v_up-v_dn)/2 e_x and -(...)/2 e_y.
__global__ void from_frame(int size,const double* v,const double* e,double* potential) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=gridDim.x*blockDim.x) {
        const double sum=.5*(v[i]+v[i+size]),difference=.5*(v[i]-v[i+size]);
        potential[i]=sum+difference*e[i+2*size];potential[i+size]=sum-difference*e[i+2*size];
        potential[i+2*size]=difference*e[i];potential[i+3*size]=-difference*e[i+size];
    }
}
// Multicollinear. With s_e = e.m and Ebar(l) = avg_e E_col[n, l s_e], E = d/dl [l Ebar] at l = 1;
// V_n = d/dl [l avg_e V_n,e] and V_m = d/dl [l^2 avg_e e V_s,e], from the one-sided stencil
// at l = 1, 1-h, 1-2h, exactly as multicollinear in remag_xc.F90.in.
constexpr double mc_h=1.e-4;
__global__ void mc_begin(int size,const double* d,double* base,double* accumulated,double* sums,
                         const PbeProblem* outer,PbeProblem* inner) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=gridDim.x*blockDim.x) {
        base[i]=d[i]+d[i+size];base[i+size]=2.*d[i+2*size];base[i+2*size]=-2.*d[i+3*size];base[i+3*size]=d[i]-d[i+size];
        for(int k=0;k<5;k++) accumulated[i+size*k]=0.;
    }
    if(!blockIdx.x&&!threadIdx.x) {*inner=*outer;inner->spins=outer->spins==4?2:0;for(int k=0;k<12;k++) sums[k]=0.;}
}
__global__ void mc_channels(int size,const double* base,const double* e,double lambda,double* channels) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=gridDim.x*blockDim.x) {
        const double s=e[0]*base[i+size]+e[1]*base[i+2*size]+e[2]*base[i+3*size];
        channels[i]=.5*(base[i]+lambda*s);channels[i+size]=.5*(base[i]-lambda*s);
    }
}
// accumulated: V_n, V_m (3), and V_s s at l = 1 (for the energy), all weighted.
__global__ void mc_accumulate(int size,const double* base,const double* e,double w,const double* v,double* accumulated) {
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=gridDim.x*blockDim.x) {
        const double* v1=v;const double* v2=v+2*size;const double* v3=v+4*size;
        const double vn0=.5*(v1[i]+v1[i+size]),vs0=.5*(v1[i]-v1[i+size]);
        const double dvn=(3.*vn0-2.*(v2[i]+v2[i+size])+.5*(v3[i]+v3[i+size]))/(2.*mc_h);
        const double dvs=(3.*vs0-2.*(v2[i]-v2[i+size])+.5*(v3[i]-v3[i+size]))/(2.*mc_h);
        const double s=e[0]*base[i+size]+e[1]*base[i+2*size]+e[2]*base[i+3*size];
        accumulated[i]+=w*(vn0+dvn);
        for(int k=0;k<3;k++) accumulated[i+size*(1+k)]+=w*e[k]*(2.*vs0+dvs);
        accumulated[i+4*size]+=w*vs0*s;
    }
}
// sums: Ex+Ec of the l = 1 calls, Ec, Dc, stress (9).
__global__ void mc_scalars(const PbeResult* r,double w,double* sums,PbeResult* out) {
    for(int i=0;i<3;i++) if(r[i].status) atomicMax(&out->status,r[i].status);
    sums[0]+=w*(r[0].energies[0]+r[0].energies[1]);sums[1]+=w*r[0].energies[1];sums[2]+=w*r[0].energies[3];
    for(int k=0;k<9;k++) sums[3+k]+=w*(r[0].stress[k]+(3.*r[0].stress[k]-4.*r[1].stress[k]+r[2].stress[k])/(2.*mc_h));
}
__global__ void mc_potential(int size,const double* base,const double* accumulated,double* potential,double* partial) {
    __shared__ double sums[2][threads];double value[2]{};
    for(int i=blockIdx.x*blockDim.x+threadIdx.x;i<size;i+=gridDim.x*blockDim.x) {
        const double vn=accumulated[i],vx=accumulated[i+size],vy=accumulated[i+2*size],vz=accumulated[i+3*size];
        potential[i]=vn+vz;potential[i+size]=vn-vz;potential[i+2*size]=vx;potential[i+3*size]=-vy;
        value[0]+=base[i]*vn+base[i+size]*vx+base[i+2*size]*vy+base[i+3*size]*vz;
        value[1]+=accumulated[i+4*size];
    }
    for(int f=0;f<2;f++) sums[f][threadIdx.x]=value[f];
    __syncthreads();
    for(int step=threads/2;step;step/=2) {
        if(threadIdx.x<step) for(int f=0;f<2;f++) sums[f][threadIdx.x]+=sums[f][threadIdx.x+step];
        __syncthreads();
    }
    if(threadIdx.x==0) for(int f=0;f<2;f++) partial[2*blockIdx.x+f]=sums[f][0];
}
__global__ void mc_finish(int size,int blocks,const PbeProblem* p,const double* partial,const double* sums,PbeResult* out) {
    if(out->status) return;
    double nv=0.,vss=0.;for(int i=0;i<blocks;i++) {nv+=partial[2*i];vss+=partial[2*i+1];}
    const double dvol=p->volume/size,energy=sums[0]+dvol*vss;
    out->energies[1]=sums[1];out->energies[0]=energy-sums[1];out->energies[3]=sums[2];
    out->energies[2]=energy-dvol*nv-sums[2];
    for(int k=0;k<9;k++) out->stress[k]=sums[3+k];
    for(int k=0;k<4;k++) if(!isfinite(out->energies[k])) out->status=2;
    for(int k=0;k<9;k++) if(!isfinite(out->stress[k])) out->status=2;
}
__global__ void reduce(int size,int blocks,const PbeProblem* p,const double* partial,PbeResult* out) {
    __shared__ double sums[fields][threads];
    for(int field=0;field<fields;field++) {
        double value=0.;for(int i=threadIdx.x;i<blocks;i+=threads) value+=partial[fields*i+field];
        sums[field][threadIdx.x]=value;
    }
    __syncthreads();
    for(int step=threads/2;step;step/=2) {
        if(threadIdx.x<step) for(int field=0;field<fields;field++) sums[field][threadIdx.x]+=sums[field][threadIdx.x+step];
        __syncthreads();
    }
    if(threadIdx.x==0) {
        for(int field=0;field<fields;field++) if(!isfinite(sums[field][0])) out->status=2;
        if(out->status) return;
        for(int i=0;i<4;i++) out->energies[i]=2*p->volume/size*sums[i][0];
        for(int i=0;i<9;i++) out->stress[i]=2./size*sums[4+i][0];
        for(int i=0;i<3;i++) out->stress[4*i]+=(out->energies[0]+out->energies[1])/p->volume;
        for(int i=0;i<4;i++) if(!isfinite(out->energies[i])) out->status=2;
        for(int i=0;i<9;i++) if(!isfinite(out->stress[i])) out->status=2;
    }
}
}

namespace {
int frame_mode=0;
Frame* frame_state=nullptr;                  // device, allocated by the first local-frame plan
Frame host_frame{};                          // published by the host before any plan existed
void frame_allocate() {
    if(frame_state) return;
    check(cudaMalloc(reinterpret_cast<void**>(&frame_state),sizeof(Frame)));
    check(cudaMemcpy(frame_state,&host_frame,sizeof(Frame),cudaMemcpyHostToDevice));
}
int mc_polar=4,mc_azimuth=-1;
// The half-sphere product rule of mc_directions in remag_xc.F90.in: the polar
// Gauss-Legendre roots with cos(theta) > 0 of P_(2 polar), by the same Newton
// iteration, times uniform azimuths; weights sum to one.
void mc_rule(int polar,int azimuth,std::vector<double>& directions,std::vector<double>& weights) {
    if(polar<1) throw std::runtime_error("REMAG.XC.Multicollinear.Polar must be positive");
    const int na=azimuth<0?4*polar:azimuth,nt=2*polar;
    if(na<1) throw std::runtime_error("REMAG.XC.Multicollinear.Azimuth must be positive");
    const double pi=3.14159265358979323846;
    for(int i=1;i<=polar;i++) {
        double x=std::cos(pi*(i-.25)/(nt+.5)),p0=1.,p1=x,dp1=0.;
        for(int it=1;it<=100;it++) {
            p0=1.;p1=x;
            for(int j=2;j<=nt;j++) {const double p2=((2*j-1)*x*p1-(j-1)*p0)/j;p0=p1;p1=p2;}
            dp1=nt*(x*p1-p0)/(x*x-1.);
            if(std::fabs(p1/dp1)<1.e-15) break;
            x-=p1/dp1;
        }
        const double w=2./((1.-x*x)*dp1*dp1);
        for(int j=1;j<=na;j++) {
            const double phi=2.*pi*(j-1)/na,r=std::sqrt(1.-x*x);
            directions.insert(directions.end(),{r*std::cos(phi),r*std::sin(phi),x});weights.push_back(w/na);
        }
    }
}
}
extern "C" void remag_set_xc_frame(int mode) {frame_mode=mode==1||mode==2?mode:0;}
extern "C" void remag_set_xc_multicollinear(int polar,int azimuth) {mc_polar=polar;mc_azimuth=azimuth;}
extern "C" int remag_xc_frame_mode() {return frame_mode;}
extern "C" void remag_query_xc_frame(int* decided,int* signed_axis,double* axis) {
    Frame f=host_frame;
    if(frame_state&&cudaMemcpy(&f,frame_state,sizeof(Frame),cudaMemcpyDeviceToHost)!=cudaSuccess) f=host_frame;
    *decided=f.decided;*signed_axis=f.signed_axis;for(int a=0;a<3;a++) axis[a]=f.axis[a];
}
extern "C" void remag_publish_xc_frame(int signed_axis,const double* axis) {
    host_frame={1,signed_axis?1:0,{axis[0],axis[1],axis[2]}};
    if(frame_state) cudaMemcpy(frame_state,&host_frame,sizeof(Frame),cudaMemcpyHostToDevice);
}
extern "C" void remag_reset_xc_frame() {
    host_frame=Frame{};if(frame_state) cudaMemcpy(frame_state,&host_frame,sizeof(Frame),cudaMemcpyHostToDevice);
}
PbePlan::PbePlan(int n1,int n2,int n3,int spins,cudaStream_t stream):
    mesh_{n1,n2,n3},spins_(spins),stream_(stream) {
    if((spins!=1&&spins!=2&&spins!=4)||n1<1||n2<1||n3<1||
       static_cast<long long>(n1)*n2>INT_MAX/n3||n1*n2*n3>INT_MAX/spins)
        throw std::runtime_error("Invalid PBE grid shape or spin count");
    size_=n1*n2*n3;blocks_=std::min(1024,(size_-1)/threads+1);frame_=spins==4?frame_mode:0;
    try {
        check(cudaMalloc(reinterpret_cast<void**>(&points_),size_t(size_)*sizeof(PbePoint)));
        check(cudaMalloc(reinterpret_cast<void**>(&partial_),size_t(blocks_)*fields*sizeof(double)));
        if(frame_==2) {
            std::vector<double> directions,weights;mc_rule(mc_polar,mc_azimuth,directions,weights);
            mc_count_=int(weights.size());
            check(cudaMalloc(reinterpret_cast<void**>(&mc_directions_),directions.size()*sizeof(double)));
            check(cudaMemcpy(mc_directions_,directions.data(),directions.size()*sizeof(double),cudaMemcpyHostToDevice));
            const size_t bytes=size_t(size_)*sizeof(double);
            check(cudaMalloc(reinterpret_cast<void**>(&base_),4*bytes));
            check(cudaMalloc(reinterpret_cast<void**>(&channels_),2*bytes));
            check(cudaMalloc(reinterpret_cast<void**>(&stencil_),6*bytes));
            check(cudaMalloc(reinterpret_cast<void**>(&accumulated_),5*bytes));
            check(cudaMalloc(reinterpret_cast<void**>(&sums_),12*sizeof(double)));
            check(cudaMalloc(reinterpret_cast<void**>(&moments_),size_t(blocks_)*2*sizeof(double)));
            check(cudaMalloc(reinterpret_cast<void**>(&stencil_results_),3*sizeof(PbeResult)));
            check(cudaMalloc(reinterpret_cast<void**>(&channel_problem_),sizeof(PbeProblem)));
            host_weights_=weights;
        } else if(frame_) {
            frame_allocate();
            const size_t bytes=size_t(size_)*sizeof(double);
            check(cudaMalloc(reinterpret_cast<void**>(&channels_),2*bytes));
            check(cudaMalloc(reinterpret_cast<void**>(&channel_potential_),2*bytes));
            check(cudaMalloc(reinterpret_cast<void**>(&directions_),3*bytes));
            check(cudaMalloc(reinterpret_cast<void**>(&moments_),size_t(blocks_)*12*sizeof(double)));
            check(cudaMalloc(reinterpret_cast<void**>(&channel_problem_),sizeof(PbeProblem)));
        }
    } catch(...) {release();throw;}
}
void PbePlan::release() {
    void* buffers[]={points_,partial_,channels_,channel_potential_,directions_,moments_,channel_problem_,
                     mc_directions_,base_,stencil_,accumulated_,sums_,stencil_results_};
    for(void* p:buffers) if(p) cudaFree(p);
}
PbePlan::~PbePlan() {release();}
size_t PbePlan::workspace_bytes() const {
    return size_t(size_)*sizeof(PbePoint)+size_t(blocks_)*fields*sizeof(double)+
           (frame_==1?size_t(size_)*7*sizeof(double)+size_t(blocks_)*12*sizeof(double)+sizeof(PbeProblem):0)+
           (frame_==2?size_t(size_)*17*sizeof(double)+size_t(blocks_)*2*sizeof(double)+3*sizeof(PbeResult)+
                      sizeof(PbeProblem)+(12+3*size_t(mc_count_))*sizeof(double):0);
}
int PbePlan::apply(const PbeProblem* problem,const double* density,double* potential,PbeResult* result) {
    if(!problem||!density||!potential||!result||density==potential) return int(cudaErrorInvalidValue);
    if(frame_==2) return apply_multicollinear(problem,density,potential,result);
    if(frame_) {
        // Collinear PBE on the two local-frame channels, then the chain rule.
        frame_problem<<<1,1,0,stream_>>>(problem,channel_problem_);
        initialize<<<1,1,0,stream_>>>(channel_problem_,result,mesh_[0],mesh_[1],mesh_[2],2);
        frame_moments<<<blocks_,threads,0,stream_>>>(size_,density,frame_state,moments_);
        frame_decide<<<1,1,0,stream_>>>(blocks_,moments_,frame_state);
        to_frame<<<blocks_,threads,0,stream_>>>(size_,density,frame_state,channels_,directions_,result);
        evaluate<<<blocks_,threads,0,stream_>>>(size_,2,channel_problem_,channels_,points_,result);
        assemble<<<blocks_,threads,0,stream_>>>(size_,2,channel_problem_,channels_,points_,channel_potential_,partial_,result);
        reduce<<<1,threads,0,stream_>>>(size_,blocks_,channel_problem_,partial_,result);
        from_frame<<<blocks_,threads,0,stream_>>>(size_,channel_potential_,directions_,potential);
        return int(cudaGetLastError());
    }
    initialize<<<1,1,0,stream_>>>(problem,result,mesh_[0],mesh_[1],mesh_[2],spins_);
    evaluate<<<blocks_,threads,0,stream_>>>(size_,spins_,problem,density,points_,result);
    assemble<<<blocks_,threads,0,stream_>>>(size_,spins_,problem,density,points_,potential,partial_,result);
    reduce<<<1,threads,0,stream_>>>(size_,blocks_,problem,partial_,result);
    return int(cudaGetLastError());
}
int PbePlan::apply_multicollinear(const PbeProblem* problem,const double* density,double* potential,PbeResult* result) {
    // The outer result carries status only until mc_finish writes the sums.
    initialize<<<1,1,0,stream_>>>(problem,result,mesh_[0],mesh_[1],mesh_[2],4);
    mc_begin<<<blocks_,threads,0,stream_>>>(size_,density,base_,accumulated_,sums_,problem,channel_problem_);
    const double lambda[3]={1.,1.-mc_h,1.-2.*mc_h};
    for(int o=0;o<mc_count_;o++) {
        const double* e=mc_directions_+3*o;const double w=host_weights_[o];
        for(int i=0;i<3;i++) {
            mc_channels<<<blocks_,threads,0,stream_>>>(size_,base_,e,lambda[i],channels_);
            PbeResult* r=stencil_results_+i;
            initialize<<<1,1,0,stream_>>>(channel_problem_,r,mesh_[0],mesh_[1],mesh_[2],2);
            evaluate<<<blocks_,threads,0,stream_>>>(size_,2,channel_problem_,channels_,points_,r);
            assemble<<<blocks_,threads,0,stream_>>>(size_,2,channel_problem_,channels_,points_,stencil_+2*size_t(size_)*i,partial_,r);
            reduce<<<1,threads,0,stream_>>>(size_,blocks_,channel_problem_,partial_,r);
        }
        mc_accumulate<<<blocks_,threads,0,stream_>>>(size_,base_,e,w,stencil_,accumulated_);
        mc_scalars<<<1,1,0,stream_>>>(stencil_results_,w,sums_,result);
    }
    mc_potential<<<blocks_,threads,0,stream_>>>(size_,base_,accumulated_,potential,moments_);
    mc_finish<<<1,1,0,stream_>>>(size_,blocks_,problem,moments_,sums_,result);
    return int(cudaGetLastError());
}
}
