// SPDX-License-Identifier: GPL-3.0-or-later
// Capability experiment only: can the pinned solver execute in a device-controlled loop?
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <complex>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace {
void check(cudaError_t code) {
    if(code!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(code));
}
void check(cusolverStatus_t code) {
    if(code!=CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("cuSOLVER status "+std::to_string(int(code)));
}
void check(cublasStatus_t code) {
    if(code!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cuBLAS status "+std::to_string(int(code)));
}
struct Context {
    cudaStream_t stream=nullptr;
    cusolverDnHandle_t solver=nullptr;
    cublasHandle_t blas=nullptr;
    syevjInfo_t jacobi=nullptr;
    cusolverDnParams_t params=nullptr;
    cudaGraph_t captured=nullptr,outer=nullptr;
    cudaGraphExec_t executable=nullptr;
    std::vector<void*> memory;
    ~Context() {
        if(executable) cudaGraphExecDestroy(executable);
        if(outer) cudaGraphDestroy(outer);
        if(captured) cudaGraphDestroy(captured);
        if(jacobi) cusolverDnDestroySyevjInfo(jacobi);
        if(params) cusolverDnDestroyParams(params);
        if(solver) cusolverDnDestroy(solver);
        if(blas) cublasDestroy(blas);
        if(stream) cudaStreamDestroy(stream);
        for(void* p:memory) cudaFree(p);
    }
    template<class T> T* allocate(size_t count) {
        void* p=nullptr;check(cudaMalloc(&p,count*sizeof(T)));memory.push_back(p);return static_cast<T*>(p);
    }
};
struct Result {
    bool warmed=false,captured=false,inspected=false,loop=false,measured=false,numerical=false,iterations_read=false;
    int nodes=0,host=0,h2d=0,d2h=0,default_copy=0,iterations=0;
    long long solver_host_workspace=-1,solver_device_workspace=-1;
    double residual=0.,norm_error=0.,orthogonality=0.;
    std::string stage="setup",operation,error;
};
void inspect(cudaGraph_t graph,Result& r) {
    size_t count=0;check(cudaGraphGetNodes(graph,nullptr,&count));
    std::vector<cudaGraphNode_t> nodes(count);check(cudaGraphGetNodes(graph,nodes.data(),&count));
    for(auto node:nodes) {
        cudaGraphNodeType type;check(cudaGraphNodeGetType(node,&type));r.nodes++;
        if(type==cudaGraphNodeTypeHost) r.host++;
        if(type==cudaGraphNodeTypeMemcpy) {
            cudaMemcpy3DParms p{};check(cudaGraphMemcpyNodeGetParams(node,&p));
            if(p.kind==cudaMemcpyHostToDevice) r.h2d++;
            if(p.kind==cudaMemcpyDeviceToHost) r.d2h++;
            if(p.kind==cudaMemcpyDefault) r.default_copy++;
        }
        if(type==cudaGraphNodeTypeGraph) {
            cudaGraph_t child;check(cudaGraphChildGraphNodeGetGraph(node,&child));inspect(child,r);
        }
    }
}
__global__ void next_iteration(cudaGraphConditionalHandle handle,int* iterations,const int* info) {
    ++*iterations;
    cudaGraphSetConditional(handle,(*info==0&&*iterations<2)?1:0);
}
__global__ void combine_status(int n,const int* cholesky,int* eigen) {
    if(*cholesky) *eigen=*cholesky>0?n+*cholesky:*cholesky;
}
template<class T> __global__ void perturb_hamiltonian(int n,const int* iterations,T* h) {
    int row=blockIdx.x*blockDim.x+threadIdx.x;
    if(row>=n) return;
    // A different problem on the second device iteration detects stale host
    // work baked into a graph. This perturbation changes eigenvectors as well
    // as eigenvalues; it is not a uniform spectral shift.
    double delta=*iterations*.2*(1.+double(row%7)/7.)*(1.+.002*row/n);
    if constexpr(std::is_same_v<T,double>) h[size_t(row)*n+row]+=delta;
    else h[size_t(row)*n+row].x+=delta;
}
template<class T> __global__ void weight_overlap(int n,const T* z,T* weighted) {
    size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i>=size_t(n)*n) return;
    double s=1.+.002*(i%n)/n;
    if constexpr(std::is_same_v<T,double>) weighted[i]=s*z[i];
    else weighted[i]=make_cuDoubleComplex(s*z[i].x,s*z[i].y);
}
template<class T> T value(double re,double im=0.) {
    if constexpr(std::is_same_v<T,double>) return re;
    else return make_cuDoubleComplex(re,im);
}
template<class T> std::complex<double> complex_value(T x) {
    if constexpr(std::is_same_v<T,double>) return {x,0.};
    else return {x.x,x.y};
}
template<class T> Result probe(int n,int algorithm) {
    Result r;
    try {
        bool use_jacobi=algorithm==1||algorithm==2,reduced=algorithm>=2,batched=algorithm==3;
        Context c;
        check(cudaStreamCreateWithFlags(&c.stream,cudaStreamNonBlocking));check(cusolverDnCreate(&c.solver));
        check(cusolverDnSetStream(c.solver,c.stream));
        check(cublasCreate(&c.blas));check(cublasSetStream(c.blas,c.stream));
        check(cublasSetPointerMode(c.blas,CUBLAS_POINTER_MODE_DEVICE));
        if(batched) check(cusolverDnCreateParams(&c.params));
        if(use_jacobi) {
            check(cusolverDnCreateSyevjInfo(&c.jacobi));
            check(cusolverDnXsyevjSetTolerance(c.jacobi,1.e-12));
            check(cusolverDnXsyevjSetMaxSweeps(c.jacobi,100));
            check(cusolverDnXsyevjSetSortEig(c.jacobi,1));
        }
        size_t elements=size_t(n)*n,bytes=elements*sizeof(T);
        std::vector<T> h(elements,value<T>(0.)),s(elements,value<T>(0.)),z(elements);
        std::vector<double> w(n);
        constexpr double imaginary=(std::is_same_v<T,double>) ? 0.0 : 0.04;
        for(int i=0;i<n;i++) {
            h[size_t(i)*n+i]=value<T>(10.+.01*i);
            s[size_t(i)*n+i]=value<T>(1.+.002*i/n);
            if(i+1<n) {
                h[size_t(i)*n+i+1]=value<T>(.125,imaginary);
                h[size_t(i+1)*n+i]=value<T>(.125,-imaginary);
            }
        }
        T* original_h=c.allocate<T>(elements);T* original_s=c.allocate<T>(elements);
        T* a=c.allocate<T>(elements);T* b=c.allocate<T>(elements);
        double* dw=c.allocate<double>(n);int* info=c.allocate<int>(1);int* iterations=c.allocate<int>(1);
        int* chol_info=c.allocate<int>(1);
        T one=value<T>(1.),zero=value<T>(0.);T* alpha=c.allocate<T>(1);T* beta=c.allocate<T>(1);
        check(cudaMemcpy(alpha,&one,sizeof(T),cudaMemcpyHostToDevice));
        check(cudaMemcpy(beta,&zero,sizeof(T),cudaMemcpyHostToDevice));
        check(cudaMemcpy(original_h,h.data(),bytes,cudaMemcpyHostToDevice));
        check(cudaMemcpy(original_s,s.data(),bytes,cudaMemcpyHostToDevice));
        int workspace=0;
        if constexpr(std::is_same_v<T,double>) {
            if(reduced) {
                int chol_work=0;
                check(cusolverDnDpotrf_bufferSize(c.solver,CUBLAS_FILL_MODE_LOWER,n,b,n,&chol_work));
                if(!batched) check(cusolverDnDsyevj_bufferSize(c.solver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,a,n,dw,&workspace,c.jacobi));
                workspace=std::max(workspace,chol_work);
            } else if(use_jacobi) check(cusolverDnDsygvj_bufferSize(c.solver,CUSOLVER_EIG_TYPE_1,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,a,n,b,n,dw,&workspace,c.jacobi));
            else check(cusolverDnDsygvd_bufferSize(c.solver,CUSOLVER_EIG_TYPE_1,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,a,n,b,n,dw,&workspace));
        } else {
            if(reduced) {
                int chol_work=0;
                check(cusolverDnZpotrf_bufferSize(c.solver,CUBLAS_FILL_MODE_LOWER,n,b,n,&chol_work));
                if(!batched) check(cusolverDnZheevj_bufferSize(c.solver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,a,n,dw,&workspace,c.jacobi));
                workspace=std::max(workspace,chol_work);
            } else if(use_jacobi) check(cusolverDnZhegvj_bufferSize(c.solver,CUSOLVER_EIG_TYPE_1,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,a,n,b,n,dw,&workspace,c.jacobi));
            else check(cusolverDnZhegvd_bufferSize(c.solver,CUSOLVER_EIG_TYPE_1,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,a,n,b,n,dw,&workspace));
        }
        T* work=c.allocate<T>(workspace);
        constexpr auto datatype=(std::is_same_v<T,double>) ? CUDA_R_64F : CUDA_C_64F;
        size_t eigen_device_bytes=0,eigen_host_bytes=0;
        unsigned char* eigen_work=nullptr;
        std::vector<unsigned char> host_work;
        if(batched) {
            check(cusolverDnXsyevBatched_bufferSize(c.solver,c.params,CUSOLVER_EIG_MODE_VECTOR,
                CUBLAS_FILL_MODE_LOWER,n,datatype,a,n,CUDA_R_64F,dw,datatype,&eigen_device_bytes,&eigen_host_bytes,1));
            if(eigen_device_bytes) eigen_work=c.allocate<unsigned char>(eigen_device_bytes);
            host_work.resize(eigen_host_bytes);
            r.solver_host_workspace=static_cast<long long>(eigen_host_bytes);
            r.solver_device_workspace=static_cast<long long>(eigen_device_bytes);
        }
        auto triangular_solve=[&](cublasSideMode_t side,cublasOperation_t operation) {
            if constexpr(std::is_same_v<T,double>)
                check(cublasDtrsm(c.blas,side,CUBLAS_FILL_MODE_LOWER,operation,CUBLAS_DIAG_NON_UNIT,n,n,alpha,b,n,a,n));
            else
                check(cublasZtrsm(c.blas,side,CUBLAS_FILL_MODE_LOWER,operation,CUBLAS_DIAG_NON_UNIT,n,n,alpha,b,n,a,n));
        };
        auto solve=[&]() -> cusolverStatus_t {
            if(reduced) {
                r.operation="Cholesky";
                if constexpr(std::is_same_v<T,double>)
                    check(cusolverDnDpotrf(c.solver,CUBLAS_FILL_MODE_LOWER,n,b,n,work,workspace,chol_info));
                else
                    check(cusolverDnZpotrf(c.solver,CUBLAS_FILL_MODE_LOWER,n,b,n,work,workspace,chol_info));
                constexpr auto adjoint=(std::is_same_v<T,double>) ? CUBLAS_OP_T : CUBLAS_OP_C;
                r.operation="two-sided triangular reduction";
                triangular_solve(CUBLAS_SIDE_LEFT,CUBLAS_OP_N);
                triangular_solve(CUBLAS_SIDE_RIGHT,adjoint);
                r.operation=batched?"standard batched eigensolve":"standard Jacobi eigensolve";
                if(batched) {
                    check(cusolverDnXsyevBatched(c.solver,c.params,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,
                        n,datatype,a,n,CUDA_R_64F,dw,datatype,eigen_work,eigen_device_bytes,
                        host_work.empty()?nullptr:host_work.data(),eigen_host_bytes,info,1));
                } else {
                    if constexpr(std::is_same_v<T,double>)
                        check(cusolverDnDsyevj(c.solver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,a,n,dw,work,workspace,info,c.jacobi));
                    else
                        check(cusolverDnZheevj(c.solver,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,a,n,dw,work,workspace,info,c.jacobi));
                }
                r.operation="eigenvector back-transform";
                triangular_solve(CUBLAS_SIDE_LEFT,adjoint);
                combine_status<<<1,1,0,c.stream>>>(n,chol_info,info);check(cudaGetLastError());
                return CUSOLVER_STATUS_SUCCESS;
            }
            r.operation="generalized eigensolve";
            if constexpr(std::is_same_v<T,double>) {
                if(use_jacobi) return cusolverDnDsygvj(c.solver,CUSOLVER_EIG_TYPE_1,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,a,n,b,n,dw,work,workspace,info,c.jacobi);
                return cusolverDnDsygvd(c.solver,CUSOLVER_EIG_TYPE_1,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,a,n,b,n,dw,work,workspace,info);
            } else {
                if(use_jacobi) return cusolverDnZhegvj(c.solver,CUSOLVER_EIG_TYPE_1,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,a,n,b,n,dw,work,workspace,info,c.jacobi);
                return cusolverDnZhegvd(c.solver,CUSOLVER_EIG_TYPE_1,CUSOLVER_EIG_MODE_VECTOR,CUBLAS_FILL_MODE_LOWER,n,a,n,b,n,dw,work,workspace,info);
            }
        };
        r.stage="warmup";
        check(cudaMemsetAsync(iterations,0,sizeof(int),c.stream));
        check(cudaMemcpyAsync(a,original_h,bytes,cudaMemcpyDeviceToDevice,c.stream));
        check(cudaMemcpyAsync(b,original_s,bytes,cudaMemcpyDeviceToDevice,c.stream));
        check(solve());check(cudaStreamSynchronize(c.stream));
        int warmup_status=0;
        check(cudaMemcpy(&warmup_status,info,sizeof(int),cudaMemcpyDeviceToHost));
        if(warmup_status!=0) throw std::runtime_error("warmup solver info "+std::to_string(warmup_status));
        r.warmed=true;
        r.stage="capture";
        check(cudaStreamBeginCapture(c.stream,cudaStreamCaptureModeGlobal));
        auto copy_h=cudaMemcpyAsync(a,original_h,bytes,cudaMemcpyDeviceToDevice,c.stream);
        auto copy_s=cudaMemcpyAsync(b,original_s,bytes,cudaMemcpyDeviceToDevice,c.stream);
        perturb_hamiltonian<<<(n+255)/256,256,0,c.stream>>>(n,iterations,a);
        auto perturb_status=cudaGetLastError();
        cusolverStatus_t solved=CUSOLVER_STATUS_SUCCESS;
        try { solved=solve(); }
        catch(...) { cudaStreamEndCapture(c.stream,&c.captured);throw; }
        auto ended=cudaStreamEndCapture(c.stream,&c.captured);
        check(copy_h);check(copy_s);check(perturb_status);check(solved);check(ended);r.captured=true;
        r.stage="inspection";inspect(c.captured,r);r.inspected=true;
        r.stage="conditional_graph";
        check(cudaGraphCreate(&c.outer,0));
        cudaGraphConditionalHandle handle;
        check(cudaGraphConditionalHandleCreate(&handle,c.outer,1,cudaGraphCondAssignDefault));
        cudaGraphNodeParams conditional{};conditional.type=cudaGraphNodeTypeConditional;
        conditional.conditional.handle=handle;conditional.conditional.type=cudaGraphCondTypeWhile;
        conditional.conditional.size=1;
        cudaGraphNode_t condition,child,finish;
        check(cudaGraphAddNode(&condition,c.outer,nullptr,0,&conditional));
        cudaGraph_t body=conditional.conditional.phGraph_out[0];
        check(cudaGraphAddChildGraphNode(&child,body,nullptr,0,c.captured));
        void* arguments[]={&handle,&iterations,&info};
        cudaKernelNodeParams kernel{};kernel.func=reinterpret_cast<void*>(next_iteration);
        kernel.gridDim=dim3(1);kernel.blockDim=dim3(1);kernel.kernelParams=arguments;
        check(cudaGraphAddKernelNode(&finish,body,&child,1,&kernel));
        check(cudaGraphInstantiate(&c.executable,c.outer,0));
        check(cudaMemsetAsync(iterations,0,sizeof(int),c.stream));
        r.stage="launch";check(cudaGraphLaunch(c.executable,c.stream));check(cudaStreamSynchronize(c.stream));
        int status=0;
        check(cudaMemcpy(&status,info,sizeof(int),cudaMemcpyDeviceToHost));
        check(cudaMemcpy(&r.iterations,iterations,sizeof(int),cudaMemcpyDeviceToHost));
        r.iterations_read=true;
        if(status!=0) throw std::runtime_error("device solver info "+std::to_string(status));
        r.loop=r.iterations==2;
        check(cudaMemcpy(z.data(),a,bytes,cudaMemcpyDeviceToHost));
        check(cudaMemcpy(w.data(),dw,size_t(n)*sizeof(double),cudaMemcpyDeviceToHost));
        r.stage="validation";bool finite=true;
        for(int col=0;col<n;col++) {
            double norm=0.;
            for(int row=0;row<n;row++) {
                auto zi=complex_value(z[size_t(col)*n+row]);double overlap=1.+.002*row/n;
                double delta=(r.iterations-1)*.2*(1.+double(row%7)/7.)*overlap;
                auto hz=(10.+.01*row+delta)*zi;
                if(row) hz+=std::complex<double>(.125,imaginary)*complex_value(z[size_t(col)*n+row-1]);
                if(row+1<n) hz+=std::complex<double>(.125,-imaginary)*complex_value(z[size_t(col)*n+row+1]);
                double error=std::abs(hz-overlap*w[col]*zi)/std::max(1.,std::abs(w[col]));
                finite=finite&&std::isfinite(error)&&std::isfinite(w[col]);
                r.residual=std::max(r.residual,error);norm+=overlap*std::norm(zi);
            }
            finite=finite&&std::isfinite(norm);r.norm_error=std::max(r.norm_error,std::abs(norm-1.));
        }
        // Full Z^H S Z validation occurs after the captured loop. The test
        // overlap is diagonal; use an independent GEMM and inspect every entry.
        T* weighted=c.allocate<T>(elements);T* gram=c.allocate<T>(elements);
        weight_overlap<<<(elements+255)/256,256,0,c.stream>>>(n,a,weighted);check(cudaGetLastError());
        if constexpr(std::is_same_v<T,double>)
            check(cublasDgemm(c.blas,CUBLAS_OP_T,CUBLAS_OP_N,n,n,n,alpha,a,n,weighted,n,beta,gram,n));
        else
            check(cublasZgemm(c.blas,CUBLAS_OP_C,CUBLAS_OP_N,n,n,n,alpha,a,n,weighted,n,beta,gram,n));
        check(cudaStreamSynchronize(c.stream));
        check(cudaMemcpy(z.data(),gram,bytes,cudaMemcpyDeviceToHost));
        for(int col=0;col<n;col++) for(int row=0;row<n;row++) {
            double error=std::abs(complex_value(z[size_t(col)*n+row])-(row==col?1.:0.));
            finite=finite&&std::isfinite(error);r.orthogonality=std::max(r.orthogonality,error);
        }
        if(!finite) throw std::runtime_error("nonfinite solver output");
        r.measured=true;r.numerical=r.residual<1.e-10&&r.norm_error<1.e-10&&r.orthogonality<1.e-10;
        r.stage="complete";
    } catch(const std::exception& e) { r.error=e.what();cudaGetLastError(); }
    return r;
}
}
int main(int argc,char** argv) {
    // One case per process: a failed stream capture can leave library-global
    // resources unusable even after destroying the handle and clearing errors.
    if(argc!=5) return 2;
    int n=std::stoi(argv[2]),complex_flag=std::stoi(argv[3]),algorithm=std::stoi(argv[4]);
    if((n!=64&&n!=576&&n!=1152)||(complex_flag!=0&&complex_flag!=1)||algorithm<0||algorithm>3) return 2;
    const char* names[]={"generalized_divide_conquer","generalized_jacobi","reduced_jacobi","reduced_batched"};
    bool complex=complex_flag!=0;
    std::ofstream out(argv[1]);if(!out) return 2;
    int devices=0;if(cudaGetDeviceCount(&devices)!=cudaSuccess||devices!=1) return 3;
    out<<std::boolalpha<<std::setprecision(12)<<"{\"cuda_build_version\":"<<CUDART_VERSION
       <<",\"scope\":\"isolated solver graph capability; not an SCF residency trace\",\"checks\":[";
    {
        Result r=complex?probe<cuDoubleComplex>(n,algorithm):probe<double>(n,algorithm);
        auto measured=[&](int v) { return r.inspected?std::to_string(v):"null"; };
        out<<"{\"n\":"<<n<<",\"complex\":"<<complex<<",\"isolated_process\":true"
           <<",\"algorithm\":"<<std::quoted(names[algorithm])
           <<",\"solver_host_workspace_bytes\":"<<(r.solver_host_workspace<0?"null":std::to_string(r.solver_host_workspace))
           <<",\"solver_device_workspace_bytes\":"<<(r.solver_device_workspace<0?"null":std::to_string(r.solver_device_workspace))
           <<",\"warmup_succeeded\":"<<r.warmed<<",\"capture_succeeded\":"<<r.captured<<",\"conditional_loop_succeeded\":"<<r.loop
           <<",\"numerical_check_passed\":"<<r.numerical<<",\"iterations\":"<<(r.iterations_read?std::to_string(r.iterations):"null")
           <<",\"captured_nodes\":"<<measured(r.nodes)<<",\"host_nodes\":"<<measured(r.host)
           <<",\"host_to_device_copy_nodes\":"<<measured(r.h2d)<<",\"device_to_host_copy_nodes\":"<<measured(r.d2h)
           <<",\"default_copy_nodes\":"<<measured(r.default_copy)<<",\"residual\":";
        if(r.measured&&std::isfinite(r.residual)) out<<r.residual;else out<<"null";
        out<<",\"column_norm_error\":";if(r.measured&&std::isfinite(r.norm_error)) out<<r.norm_error;else out<<"null";
        out<<",\"overlap_orthogonality_error\":";if(r.measured&&std::isfinite(r.orthogonality)) out<<r.orthogonality;else out<<"null";
        out<<",\"stage\":"<<std::quoted(r.stage)<<",\"operation\":"<<std::quoted(r.operation)<<",\"error\":"<<std::quoted(r.error)<<"}";
        out.flush();
    }
    out<<"]}\n";
}
