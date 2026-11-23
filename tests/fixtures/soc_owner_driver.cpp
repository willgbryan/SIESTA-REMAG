// Execute the production lease/copy function with explicit CPU transport doubles.
#include "nonlocal_h0_state.cpp"
#include <limits>
uint64_t test_epoch=2;bool test_scf=false;size_t transferred=0;int transfers=0;
namespace remag {
uint64_t geometry_epoch(){return test_epoch;}
bool in_scf(){return test_scf;}
void transfer(Transfer boundary,bool to_device,size_t bytes,size_t copies){
    if(boundary!=Transfer::soc_operator_output||to_device||copies!=1)std::abort();
    transferred+=bytes;++transfers;
}
void validate_nonlocal_h0_topology(int rows,int columns,int capacity,const int* n,const int* p,const int* l){
    require(rows==1&&columns==1&&capacity==1&&n&&p&&l&&n[0]==1&&p[0]==0&&l[0]==1,"Injected topology mismatch");
}
TwoCenterView borrow_integral_two_center(int){std::abort();}
cudaError_t expand_nonlocal_queries_device(const NonlocalQueryView&,int*,double*,int*,cudaStream_t){std::abort();}
cudaError_t evaluate_two_center_device(const TwoCenterView&,int,const int*,const double*,double*,int*,cudaStream_t){std::abort();}
cudaError_t add_nonlocal_h0_device(const NonlocalH0View&,double*,H0Complex*,int*,cudaStream_t){std::abort();}
}
extern "C" int remag_initialization_enabled(){return 1;}
extern "C" int remag_orientation_retain_geometry(){return 0;}
int main(int argc,char** argv){
    if(argc!=2)return 90;const std::string mode=argv[1];setenv("REMAG_NONLOCAL_H0","1",1);
    remag::Record r;r.epoch=2;r.rows=r.columns=r.capacity=1;r.consumer=3;
    r.production=r.bound=r.passed=r.scf_borrowed=r.scf_released=true;r.status=0;
    if(mode=="stale")r.epoch=1;
    if(mode=="scalar")r.consumer=2;
    if(mode=="unbound")r.bound=false;
    if(mode=="shadow")r.production=false;
    if(mode=="failed")r.passed=false;
    if(mode=="unleased")r.scf_borrowed=false;
    if(mode=="leased")r.scf_released=false;
    if(mode=="forces")r.force_borrowed=true;
    if(mode=="complete")r.complete=true;
    if(mode=="scf")test_scf=true;
    if(mode=="copy_failure")copy_failure=true;
    if(mode=="poison")r.poison_requested=r.host_poisoned=true;
    remag::evidence().rows.push_back(r);remag::state=std::make_unique<remag::State>();
    remag::H0Complex values[4]={{1,2},{3,4},{5,6},{7,8}},output[4];
    for(auto& v:output)v={99,98};uint64_t epoch=77;
    if(mode=="nonfinite")values[2].real=std::numeric_limits<double>::quiet_NaN();
    remag::state->z=mode=="missing"?nullptr:values;
    int n=1,p=0,l=mode=="topology"?2:1;
    const int result=remag_nonlocal_h0_soc_export(1,1,1,&n,&p,&l,output,&epoch);
    const bool success=mode=="valid"||mode=="poison"||mode=="repeat";
    if(success){
        if(result||epoch!=2||transfers!=1||transferred!=64||copy_calls!=1)return 1;
        for(int i=0;i<4;i++)if(output[i].real!=values[i].real||output[i].imag!=values[i].imag)return 2;
        if(remag::evidence().rows[0].exported||remag::evidence().rows[0].soc_export_bytes!=64)return 3;
        if(mode=="repeat"&&(!remag_nonlocal_h0_soc_export(1,1,1,&n,&p,&l,output,&epoch)||copy_calls!=1))return 4;
    }else{
        if(result!=-281||epoch!=77)return 5;
        for(auto v:output)if(v.real!=99||v.imag!=98)return 6;
        if(copy_calls!=(mode=="nonfinite"||mode=="copy_failure"?1:0))return 7;
        if(transferred!=(mode=="nonfinite"?64:0))return 8;
    }
    // State destruction must not free or overwrite a caller-owned snapshot.
    remag::state.reset();return 0;
}
