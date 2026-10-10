// SPDX-License-Identifier: GPL-3.0-or-later
// Optional CUDA 12.6 activity observer: all evidence stays in host RAM until
// bridge shutdown. Never link this into the production bridge.
#include <cupti_activity.h>
#include <cupti_version.h>
#include <cupti_callbacks.h>
#include <cupti_driver_cbid.h>
#include <cupti_runtime_cbid.h>
#include <cuda_runtime_api.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <unistd.h>

namespace {
constexpr size_t buffer_bytes=4*1024*1024, max_records=2000000;
struct Event {
    uint64_t begin=0,end=0,bytes=0;
    uint64_t graph_node=0;
    uint32_t device=0,context=0,stream=0,correlation=0,runtime_correlation=0,name=0;
    uint32_t graph=0;
    uint8_t kind=0,copy_kind=0,src_kind=0,dst_kind=0;
};
struct Phase { uint64_t timestamp; int phase,geometry,iteration; };
struct Api {
    uint64_t begin,end;
    uint32_t kind,correlation,process,thread,callback;
};
struct External { uint64_t phase; uint32_t correlation,kind; };
struct GraphLaunch { uint32_t graph,correlation,api_kind,callback,enclosing_runtime; };
struct GraphRegistration { uint32_t graph; unsigned long long flags; };
constexpr auto external_kind=CUPTI_EXTERNAL_CORRELATION_KIND_CUSTOM0;
thread_local uint64_t current_phase=0;
thread_local uint32_t runtime_graph_launch=0;
struct State {
    std::mutex mutex;
    std::vector<Event> events;
    std::vector<Phase> phases;
    std::vector<Api> apis;
    std::vector<External> external;
    std::vector<GraphLaunch> graph_launches;
    std::vector<GraphRegistration> graph_registrations;
    std::vector<std::string> names,errors;
    std::unordered_map<std::string,uint32_t> name_ids;
    std::atomic<uint64_t> dropped{0},overflow{0},callback_failures{0};
    std::atomic<bool> finished{false};
    bool active=false,memcpy_enabled=false,kernel_enabled=false,memset_enabled=false;
    bool memory_only=false;
    bool runtime_enabled=false,driver_enabled=false,external_enabled=false;
    bool graph_callbacks_enabled=false;
    CUpti_SubscriberHandle subscriber=nullptr;
    uint32_t cupti_runtime_version=0;
    uint64_t end=0;
};
// Intentionally retain state through all library destructors/CUPTI callbacks.
State& state() { static State* s=new State; return *s; }
void error(const char* operation,int status) noexcept {
    auto& s=state();
    try {
        std::lock_guard<std::mutex> lock(s.mutex);
        s.errors.push_back(std::string(operation)+": "+std::to_string(status));
    } catch(...) { s.callback_failures++; }
}
bool check(CUptiResult status,const char* operation) {
    if(status==CUPTI_SUCCESS) return true;
    error(operation,static_cast<int>(status));
    return false;
}
void CUPTIAPI graph_callback(void*,CUpti_CallbackDomain domain,CUpti_CallbackId id,const void* raw) {
    const bool driver=domain==CUPTI_CB_DOMAIN_DRIVER_API&&
        (id==CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch||id==CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz);
    const bool runtime=domain==CUPTI_CB_DOMAIN_RUNTIME_API&&
        (id==CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000||id==CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_ptsz_v10000);
    if(!driver&&!runtime) return;
    const auto* data=static_cast<const CUpti_CallbackData*>(raw);
    if(!data) {error("Missing graph callback data",-1);return;}
    if(data->callbackSite==CUPTI_API_ENTER) {
        if(runtime) {
            if(runtime_graph_launch||!data->correlationId) error("Nested or uncorrelated runtime graph launch",-1);
            runtime_graph_launch=data->correlationId;
        }
        return;
    }
    if(data->callbackSite!=CUPTI_API_EXIT) {error("Unknown graph callback site",-1);return;}
    const uint32_t enclosing_runtime=driver?runtime_graph_launch:0;
    if(runtime) {
        if(runtime_graph_launch!=data->correlationId) error("Runtime graph callback nesting mismatch",-1);
        runtime_graph_launch=0;
    }
    auto& s=state();
    try {
        if(!data->functionParams||!data->functionReturnValue||!data->correlationId) {
            error("Invalid graph launch callback",-1);return;
        }
        int result;std::memcpy(&result,data->functionReturnValue,sizeof(result));
        if(result!=0) {error("Failed graph launch",result);return;}
        // All four entry points take the graph-executable handle first.
        // Copy the parameter prefix while the callback is live; do not retain
        // parameter pointers or depend on newer generated CUDA meta headers.
        CUgraphExec executable;std::memcpy(&executable,data->functionParams,sizeof(executable));
        uint32_t graph=0;
        if(!check(cuptiGetGraphExecId(executable,&graph),"cuptiGetGraphExecId")) return;
        if(!graph) {error("Missing executable graph ID",-1);return;}
        std::lock_guard<std::mutex> lock(s.mutex);
        if(s.graph_launches.size()>=max_records) {s.overflow++;return;}
        s.graph_launches.push_back({graph,data->correlationId,
            static_cast<uint32_t>(driver?CUPTI_ACTIVITY_KIND_DRIVER:CUPTI_ACTIVITY_KIND_RUNTIME),
            static_cast<uint32_t>(id),enclosing_runtime});
    } catch(...) {s.callback_failures++;}
}
uint64_t timestamp() {
    uint64_t value=0;
    check(cuptiGetTimestamp(&value),"cuptiGetTimestamp");
    return value;
}
void CUPTIAPI request_buffer(uint8_t** buffer,size_t* size,size_t* max_records_out) {
    void* allocated=nullptr;
    if(posix_memalign(&allocated,8,buffer_bytes)!=0) state().callback_failures++;
    *buffer=static_cast<uint8_t*>(allocated);
    *size=allocated?buffer_bytes:0;
    *max_records_out=0;
}
void CUPTIAPI complete_buffer(CUcontext context,uint32_t stream,uint8_t* buffer,size_t,size_t valid) {
    auto& s=state();
    try {
        CUpti_Activity* activity=nullptr;
        while(valid>0) {
            auto status=cuptiActivityGetNextRecord(buffer,valid,&activity);
            if(status==CUPTI_ERROR_MAX_LIMIT_REACHED) break;
            if(!check(status,"cuptiActivityGetNextRecord")) break;
            Event event;
            const char* name=nullptr;
            if(activity->kind==CUPTI_ACTIVITY_KIND_MEMCPY) {
#if CUPTI_API_VERSION >= 26
                const auto& m=*reinterpret_cast<const CUpti_ActivityMemcpy6*>(activity);
                if(m.copyCount>1) error("Batched memcpy accounting unsupported",-1);
#else
                const auto& m=*reinterpret_cast<const CUpti_ActivityMemcpy5*>(activity);
#endif
                event.kind=1;event.begin=m.start;event.end=m.end;event.bytes=m.bytes;
                event.device=m.deviceId;event.context=m.contextId;event.stream=m.streamId;
                event.correlation=m.correlationId;event.runtime_correlation=m.runtimeCorrelationId;
                event.graph=m.graphId;event.graph_node=m.graphNodeId;
                event.copy_kind=m.copyKind;event.src_kind=m.srcKind;event.dst_kind=m.dstKind;
            } else if(activity->kind==CUPTI_ACTIVITY_KIND_MEMSET) {
                const auto& m=*reinterpret_cast<const CUpti_ActivityMemset4*>(activity);
                event.kind=3;event.begin=m.start;event.end=m.end;event.bytes=m.bytes;
                event.device=m.deviceId;event.context=m.contextId;event.stream=m.streamId;
                event.correlation=m.correlationId;event.dst_kind=m.memoryKind;
                event.graph=m.graphId;event.graph_node=m.graphNodeId;
            } else if(activity->kind==CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
                const auto& k=*reinterpret_cast<const CUpti_ActivityKernel9*>(activity);
                event.kind=2;event.begin=k.start;event.end=k.end;
                event.device=k.deviceId;event.context=k.contextId;event.stream=k.streamId;
                event.correlation=k.correlationId;name=k.name;
                event.graph=k.graphId;event.graph_node=k.graphNodeId;
            } else if(activity->kind==CUPTI_ACTIVITY_KIND_RUNTIME||activity->kind==CUPTI_ACTIVITY_KIND_DRIVER) {
                const auto& a=*reinterpret_cast<const CUpti_ActivityAPI*>(activity);
                std::lock_guard<std::mutex> lock(s.mutex);
                if(s.apis.size()>=max_records) { s.overflow++; continue; }
                s.apis.push_back({a.start,a.end,static_cast<uint32_t>(a.kind),a.correlationId,
                                  a.processId,a.threadId,a.cbid});
                continue;
            } else if(activity->kind==CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION) {
                const auto& e=*reinterpret_cast<const CUpti_ActivityExternalCorrelation*>(activity);
                std::lock_guard<std::mutex> lock(s.mutex);
                if(s.external.size()>=max_records) { s.overflow++; continue; }
                s.external.push_back({e.externalId,e.correlationId,static_cast<uint32_t>(e.externalKind)});
                continue;
            } else {
                error("Unexpected activity kind",static_cast<int>(activity->kind));
                continue;
            }
            std::lock_guard<std::mutex> lock(s.mutex);
            if(s.events.size()>=max_records) { s.overflow++; continue; }
            if(name) {
                auto inserted=s.name_ids.emplace(name,static_cast<uint32_t>(s.names.size()));
                if(inserted.second) s.names.emplace_back(name);
                event.name=inserted.first->second;
            } else if(event.kind==2) {
                s.callback_failures++;
            }
            s.events.push_back(event);
        }
        size_t dropped=0;
        if(check(cuptiActivityGetNumDroppedRecords(context,stream,&dropped),"cuptiActivityGetNumDroppedRecords"))
            s.dropped+=dropped;
    } catch(...) { s.callback_failures++; }
    std::free(buffer);
}
void json_string(FILE* f,const std::string& value) {
    std::fputc('"',f);
    for(unsigned char c:value) {
        if(c=='"'||c=='\\') { std::fputc('\\',f);std::fputc(c,f); }
        else if(c<32||c>=127) std::fprintf(f,"\\u%04x",static_cast<unsigned>(c));
        else std::fputc(c,f);
    }
    std::fputc('"',f);
}
unsigned long long number(uint64_t v) { return static_cast<unsigned long long>(v); }
}

extern "C" void remag_activity_graph(void* executable) {
    auto& s=state();
    if(!s.active||s.finished) return;
    // Called outside CUDA/CUPTI callbacks immediately after instantiation.
    // Flags distinguish host-only graphs from graphs that device kernels can
    // launch without a host API callback. Neither IDs nor flags assign phases.
    uint32_t graph=0;unsigned long long flags=0;
    if(!check(cuptiGetGraphExecId(static_cast<CUgraphExec>(executable),&graph),"registered executable graph ID")) return;
    auto status=cudaGraphExecGetFlags(static_cast<cudaGraphExec_t>(executable),&flags);
    if(status!=cudaSuccess) {error("registered graph flags",static_cast<int>(status));return;}
    try {
        std::lock_guard<std::mutex> lock(s.mutex);
        if(s.graph_registrations.size()>=max_records) {s.overflow++;return;}
        s.graph_registrations.push_back({graph,flags});
    } catch(...) {s.callback_failures++;}
}
extern "C" void remag_activity_phase(int phase,int geometry,int iteration) {
    auto& s=state();
    if(!s.active||s.finished) return;
    // Correlate the launching API with a region, not GPU timestamps with host
    // markers: CUPTI clock conversion can place device work before its launch.
    // External IDs are thread-local. Work from an unmarked library worker is
    // retained but must fail offline attribution instead of using a time guess.
    if(current_phase) {
        uint64_t previous=0;
        if(check(cuptiActivityPopExternalCorrelationId(external_kind,&previous),"pop phase")&&previous!=current_phase)
            error("external phase stack mismatch",-1);
        current_phase=0;
    }
    try {
        auto time=timestamp();
        uint64_t id;
        {
            std::lock_guard<std::mutex> lock(s.mutex);
            s.phases.push_back({time,phase,geometry,iteration});
            id=s.phases.size();
        }
        if(check(cuptiActivityPushExternalCorrelationId(external_kind,id),"push phase")) current_phase=id;
    } catch(...) { s.callback_failures++; }
}
extern "C" void remag_activity_finish() {
    auto& s=state();
    if(!s.active||s.finished.exchange(true)) return;
    // This synchronization is outside SCF; timing from instrumented runs is
    // never qualified. Flush alone does not ensure device activity completed.
    auto synced=cudaDeviceSynchronize();
    if(synced!=cudaSuccess) error("cudaDeviceSynchronize",static_cast<int>(synced));
    s.end=timestamp();
    if(current_phase) {
        uint64_t previous=0;
        if(check(cuptiActivityPopExternalCorrelationId(external_kind,&previous),"pop final phase")&&previous!=current_phase)
            error("external final phase stack mismatch",-1);
        current_phase=0;
    }
    if(s.memcpy_enabled) check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMCPY),"disable memcpy");
    if(s.kernel_enabled) check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),"disable kernels");
    if(s.memset_enabled) check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_MEMSET),"disable memset");
    if(s.runtime_enabled) check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_RUNTIME),"disable runtime");
    if(s.driver_enabled) check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_DRIVER),"disable driver");
    if(s.external_enabled) check(cuptiActivityDisable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION),"disable external correlation");
    if(s.subscriber) check(cuptiUnsubscribe(s.subscriber),"unsubscribe graph callbacks");
    check(cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED),"cuptiActivityFlushAll");
    size_t dropped=0;
    if(check(cuptiActivityGetNumDroppedRecords(nullptr,0,&dropped),"final dropped records")) s.dropped+=dropped;
    std::lock_guard<std::mutex> lock(s.mutex);
    FILE* f=std::fopen("remag-activity.json","w");
    if(!f) return; // Harness treats a missing trace as incomplete coverage.
    std::fprintf(f,"{\"schema_version\":2,\"cuda_build_version\":%d,\"cupti_build_version\":%d,\"cupti_runtime_version\":%u,\"memcpy_enabled\":%s,\"kernel_enabled\":%s,"
        "\"runtime_enabled\":%s,\"driver_enabled\":%s,\"external_enabled\":%s,\"external_kind\":%u,"
        "\"dropped_records\":%llu,\"record_overflow\":%llu,\"callback_failures\":%llu,\"end_ns\":%llu,\"errors\":[",
        CUDART_VERSION,CUPTI_API_VERSION,s.cupti_runtime_version,s.memcpy_enabled?"true":"false",s.kernel_enabled?"true":"false",
        s.runtime_enabled?"true":"false",s.driver_enabled?"true":"false",s.external_enabled?"true":"false",
        static_cast<unsigned>(external_kind),
        number(s.dropped),number(s.overflow),number(s.callback_failures),number(s.end));
    for(size_t i=0;i<s.errors.size();i++) { if(i) std::fputc(',',f);json_string(f,s.errors[i]); }
    std::fprintf(f,"],\"phases\":[");
    for(size_t i=0;i<s.phases.size();i++) {
        const auto& p=s.phases[i];
        std::fprintf(f,"%s{\"begin_ns\":%llu,\"phase\":%d,\"geometry\":%d,\"iteration\":%d,\"external_id\":%llu}",
            i?",":"",number(p.timestamp),p.phase,p.geometry,p.iteration,number(i+1));
    }
    std::fprintf(f,"],\"api_columns\":[\"kind\",\"correlation\",\"begin_ns\",\"end_ns\",\"process\",\"thread\",\"callback\"],\"apis\":[");
    for(size_t i=0;i<s.apis.size();i++) {
        const auto& a=s.apis[i];
        std::fprintf(f,"%s[%u,%u,%llu,%llu,%u,%u,%u]",i?",":"",a.kind,a.correlation,
            number(a.begin),number(a.end),a.process,a.thread,a.callback);
    }
    std::fprintf(f,"],\"external_columns\":[\"correlation\",\"external_kind\",\"external_id\"],\"external\":[");
    for(size_t i=0;i<s.external.size();i++) {
        const auto& e=s.external[i];
        std::fprintf(f,"%s[%u,%u,%llu]",i?",":"",e.correlation,e.kind,number(e.phase));
    }
    std::fprintf(f,"],\"kernel_names\":[");
    for(size_t i=0;i<s.names.size();i++) { if(i) std::fputc(',',f);json_string(f,s.names[i]); }
    std::fprintf(f,"],\"kernel_activity_requested\":%s,\"memset_enabled\":%s,\"graph_launch_callbacks_enabled\":%s,"
        "\"graph_launch_columns\":[\"graph_id\",\"correlation\",\"api_kind\",\"callback\",\"enclosing_runtime_correlation\"],\"graph_launches\":[",
        s.memory_only?"false":"true",s.memset_enabled?"true":"false",s.graph_callbacks_enabled?"true":"false");
    for(size_t i=0;i<s.graph_launches.size();i++) {
        const auto& g=s.graph_launches[i];
        std::fprintf(f,"%s[%u,%u,%u,%u,%u]",i?",":"",g.graph,g.correlation,g.api_kind,g.callback,g.enclosing_runtime);
    }
    std::fprintf(f,"],\"graph_registration_columns\":[\"graph_id\",\"instantiation_flags\"],\"graph_registrations\":[");
    for(size_t i=0;i<s.graph_registrations.size();i++) {
        const auto& g=s.graph_registrations[i];
        std::fprintf(f,"%s[%u,%llu]",i?",":"",g.graph,g.flags);
    }
    std::fprintf(f,"],\"events\":[");
    for(size_t i=0;i<s.events.size();i++) {
        const auto& e=s.events[i];
        // Compact, documented columns keep raw traces practical to retain.
        std::fprintf(f,"%s[%u,%llu,%llu,%llu,%u,%u,%u,%u,%u,%u,%u,%u,%u]",i?",":"",
            e.kind,number(e.begin),number(e.end),number(e.bytes),e.device,e.context,e.stream,e.correlation,
            e.runtime_correlation,e.name,e.copy_kind,e.src_kind,e.dst_kind);
    }
    // Preserve graph provenance for conditional-body events whose correlation
    // ID can be zero. These identifiers alone do not identify a replay/phase;
    // the offline parser must continue to reject uncorrelated activity.
    std::fprintf(f,"],\"event_graph_columns\":[\"graph_id\",\"graph_node_id\"],\"event_graphs\":[");
    for(size_t i=0;i<s.events.size();i++) {
        const auto& e=s.events[i];
        std::fprintf(f,"%s[%u,%llu]",i?",":"",e.graph,number(e.graph_node));
    }
    std::fprintf(f,"],\"event_columns\":[\"kind\",\"begin_ns\",\"end_ns\",\"bytes\",\"device\",\"context\",\"stream\","
        "\"correlation\",\"runtime_correlation\",\"name_id\",\"copy_kind\",\"source_kind\",\"destination_kind\"]}\n");
    std::fclose(f);
}

__attribute__((constructor)) static void initialize_activity() {
    const char* enabled=std::getenv("REMAG_TRACE_CUDA");
    const char* executable=std::getenv("REMAG_ACTIVITY_EXECUTABLE");
    if(!enabled||std::strcmp(enabled,"1")!=0||!executable) return;
    // A strace parent inherits LD_PRELOAD too; only observe the intended SIESTA
    // process, so a parent destructor cannot overwrite its child's evidence.
    char self[4096];
    auto length=readlink("/proc/self/exe",self,sizeof(self)-1);
    if(length<0) return;
    self[length]='\0';
    if(std::strcmp(self,executable)!=0) return;
    auto& s=state();
    s.active=true;
    const char* memory_only=std::getenv("REMAG_ACTIVITY_MEMORY_ONLY");
    s.memory_only=memory_only&&std::strcmp(memory_only,"1")==0;
    check(cuptiGetVersion(&s.cupti_runtime_version),"CUPTI version");
    if(s.cupti_runtime_version!=CUPTI_API_VERSION) {
        error("CUPTI build/runtime mismatch",-1);
        // Never reinterpret activity records using a mismatched ABI.
        std::atexit(remag_activity_finish);return;
    }
    check(cuptiActivityRegisterCallbacks(request_buffer,complete_buffer),"register callbacks");
    s.runtime_enabled=check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_RUNTIME),"enable runtime");
    s.driver_enabled=check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_DRIVER),"enable driver");
    s.external_enabled=check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_EXTERNAL_CORRELATION),"enable external correlation");
    if(check(cuptiSubscribe(&s.subscriber,graph_callback,nullptr),"subscribe graph callbacks")) {
        bool ordinary=check(cuptiEnableCallback(1,s.subscriber,CUPTI_CB_DOMAIN_DRIVER_API,
            CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch),"enable graph launch callback");
        bool ptsz=check(cuptiEnableCallback(1,s.subscriber,CUPTI_CB_DOMAIN_DRIVER_API,
            CUPTI_DRIVER_TRACE_CBID_cuGraphLaunch_ptsz),"enable per-thread graph launch callback");
        bool runtime=check(cuptiEnableCallback(1,s.subscriber,CUPTI_CB_DOMAIN_RUNTIME_API,
            CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_v10000),"enable runtime graph launch callback");
        bool runtime_ptsz=check(cuptiEnableCallback(1,s.subscriber,CUPTI_CB_DOMAIN_RUNTIME_API,
            CUPTI_RUNTIME_TRACE_CBID_cudaGraphLaunch_ptsz_v10000),"enable per-thread runtime graph launch callback");
        s.graph_callbacks_enabled=ordinary&&ptsz&&runtime&&runtime_ptsz;
    }
    remag_activity_phase(0,0,0);
    s.memcpy_enabled=check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMCPY),"enable memcpy");
    if(!s.memory_only) s.kernel_enabled=check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL),"enable kernels");
    s.memset_enabled=check(cuptiActivityEnable(CUPTI_ACTIVITY_KIND_MEMSET),"enable memset");
    std::atexit(remag_activity_finish);
}
