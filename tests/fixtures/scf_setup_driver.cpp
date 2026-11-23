// Real checkpoint helper, fake API results and deliberate process termination.
#include "scf_graph_setup.hpp"
#include <chrono>
#include <csignal>
#include <string>
#include <thread>
#include <sys/resource.h>

std::string mode;
int calls=0;
int step(const char* name) {
    ++calls;
    if(mode==std::string("crash-")+name)std::raise(SIGSEGV);
    return mode==std::string("error-")+name?7:0;
}
cudaError_t cudaRuntimeGetVersion(int* v) {*v=12060;return 0;}
cudaError_t cudaDriverGetVersion(int* v) {
    if(mode=="unknown-version") {*v=12345;return 9;}
    *v=mode=="unadmitted-driver"?13050:13030;return 0;
}
cudaError_t cudaGraphInstantiate(cudaGraphExec_t* exec,cudaGraph_t,unsigned long long flags) {
    if(flags)return 99;
    *exec=reinterpret_cast<void*>(2);return step("instantiate");
}
cudaError_t cudaGraphUpload(cudaGraphExec_t,cudaStream_t) {return step("upload");}
cudaError_t cudaStreamSynchronize(cudaStream_t) {return step("synchronize");}
int main(int argc,char** argv) {
    rlimit limit{0,0};setrlimit(RLIMIT_CORE,&limit);
    mode=argc>1?argv[1]:"crash-instantiate";
    if(argc==1)std::this_thread::sleep_for(std::chrono::milliseconds(700));
    cudaGraphExec_t exec=nullptr;
    auto status=remag::instantiate_scf_graph(&exec,reinterpret_cast<void*>(1),reinterpret_cast<void*>(3),
        {30,2,1896,8,120985,0});
    std::printf("%d %d\n",status,calls);
    return status;
}
