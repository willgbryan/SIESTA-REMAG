// Exercise the actual adapter/planner contract without CUDA or a solver.
#include "scf_capacity.hpp"
#include <cassert>
#include <climits>
#include <initializer_list>
int main() {
    for(int nk:{128,129,132,216,936,1728,4096}) {
        auto scalar=remag::scf_spectral_layout(1,30,2,nk);
        auto soc=remag::scf_spectral_layout(3,30,8,nk);
        assert(scalar.dimension==30&&scalar.spins==2&&scalar.jobs==2*nk);
        assert(soc.dimension==60&&soc.spins==1&&soc.jobs==nk);
        assert(soc.vector_bytes==60*60*16);
        // Last scheduled job must address its own complete retained vector
        // slot and spectrum; no prototype 128-point indexing or truncation.
        auto last=std::size_t(soc.jobs-1)*soc.vector_bytes;
        assert(last+soc.vector_bytes==std::size_t(nk)*60*60*16);
        assert((scalar.jobs-1)/scalar.spins==nk-1);
        assert((scalar.jobs-1)%scalar.spins==1);
    }
    for(int nk:{-1,0,4097,INT_MAX}) {
        assert(!remag::scf_shape_supported(1,30,2,nk));
        bool rejected=false;
        try {remag::scf_spectral_layout(3,30,8,nk);}catch(const std::runtime_error&){rejected=true;}
        assert(rejected);
    }
    assert(remag::scf_shape_supported(0,30,2,1));
    assert(!remag::scf_shape_supported(0,30,2,2));
    assert(!remag::scf_shape_supported(3,30,2,216));
    assert(!remag::scf_shape_supported(1,2049,2,132));
    auto large=remag::scf_spectral_layout(3,2048,8,4096);
    assert(large.dimension==4096);
    assert(std::size_t(large.jobs)*large.vector_bytes==1099511627776ULL);
    assert(large.dimension*large.jobs<INT_MAX);
}
