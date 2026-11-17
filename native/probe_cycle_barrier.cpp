// SPDX-License-Identifier: GPL-3.0-or-later
// Diagnostic interposer only. Never link into the production REMAG library.
// Terminates before the first complete SCF-cycle call; no fake success return.
#include <cstdio>
#include <cstdlib>
extern "C" int remag_cycle_execute(int geometry,int*,int*) {
    auto* out=std::fopen("cycle-barrier.json","w");
    if(!out)std::_Exit(91);
    const int written=std::fprintf(out,"{\"format\":\"REMAG-CYCLE-BARRIER/1\",\"geometry\":%d,\"SCF_cycle_launched\":false,\"intentional_exit\":90}\n",geometry);
    const int closed=std::fclose(out);
    std::_Exit(written<0||closed?91:90);
}
