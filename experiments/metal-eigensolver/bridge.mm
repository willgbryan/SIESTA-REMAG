// SPDX-License-Identifier: GPL-3.0-or-later
// Isolated experimental C ABI. No production dependency or build integration.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

struct Plan {
    id<MTLDevice> device;
    id<MTLCommandQueue> queue;
    id<MTLComputePipelineState> pipeline;
    id<MTLBuffer> a,v,w,status,sweeps,residual;
    unsigned n,capacity;
};
struct Params { unsigned n,count,maximum; float tolerance; };
static void error_text(char* target,size_t length,const char* message) {
    if(target && length) snprintf(target,length,"%s",message);
}
static id<MTLBuffer> buffer(id<MTLDevice> d,size_t bytes) {
    id<MTLBuffer> b=[d newBufferWithLength:bytes options:MTLResourceStorageModeShared];
    if(!b) throw std::runtime_error("Metal shared-buffer allocation failed");
    return b;
}
extern "C" void* metal_seed_create(const char* path,unsigned n,unsigned capacity,char* error,size_t size) {
    @autoreleasepool {
        try {
            if(!path || n<2 || n>128 || n%2 || !capacity || capacity>512)
                throw std::runtime_error("Expected even dimension 2..128 and batch capacity 1..512");
            auto p=std::make_unique<Plan>(); p->n=n;p->capacity=capacity;
            p->device=MTLCreateSystemDefaultDevice();
            if(!p->device) throw std::runtime_error("No Metal device visible");
            NSError* e=nil;
            NSString* source=[NSString stringWithContentsOfFile:[NSString stringWithUTF8String:path]
                encoding:NSUTF8StringEncoding error:&e];
            if(!source) throw std::runtime_error(e.localizedDescription.UTF8String);
            MTLCompileOptions* options=[MTLCompileOptions new]; options.fastMathEnabled=NO;
            id<MTLLibrary> library=[p->device newLibraryWithSource:source options:options error:&e];
            if(!library) throw std::runtime_error(e.localizedDescription.UTF8String);
            p->pipeline=[p->device newComputePipelineStateWithFunction:[library newFunctionWithName:@"jacobi"] error:&e];
            if(!p->pipeline) throw std::runtime_error(e.localizedDescription.UTF8String);
            if(p->pipeline.maxTotalThreadsPerThreadgroup<256)
                throw std::runtime_error("Device cannot run the required 256-thread group");
            p->queue=[p->device newCommandQueue];
            if(!p->queue) throw std::runtime_error("Metal command queue creation failed");
            p->a=buffer(p->device,size_t(n)*n*capacity*8); p->v=buffer(p->device,size_t(n)*n*capacity*8);
            p->w=buffer(p->device,size_t(n)*capacity*4);p->status=buffer(p->device,capacity*4);
            p->sweeps=buffer(p->device,capacity*4);p->residual=buffer(p->device,capacity*4);
            return p.release();
        } catch(const std::exception& e) {error_text(error,size,e.what());return nullptr;}
    }
}
extern "C" void metal_seed_destroy(void* handle) {delete static_cast<Plan*>(handle);}
extern "C" const char* metal_seed_device(void* handle) {
    return handle ? static_cast<Plan*>(handle)->device.name.UTF8String : "";
}
extern "C" int metal_seed_solve(void* handle,const void* matrices,unsigned count,unsigned maximum,float tolerance,
    void* vectors,float* values,int* status,unsigned* sweeps,float* residual,double* seconds,char* error,size_t size) {
    @autoreleasepool {
        try {
            auto started=std::chrono::steady_clock::now();auto p=static_cast<Plan*>(handle);
            if(!p || !matrices || !vectors || !values || !status || !sweeps || !residual || !seconds
                || !count || count>p->capacity || maximum>100 || !std::isfinite(tolerance)
                || tolerance<1e-8f || tolerance>1e-2f) throw std::runtime_error("Invalid solve arguments");
            seconds[0]=seconds[1]=NAN;
            size_t bytes=size_t(p->n)*p->n*count*8;
            memcpy(p->a.contents,matrices,bytes);
            Params cfg{p->n,count,maximum,tolerance};
            id<MTLCommandBuffer> command=[p->queue commandBuffer];
            id<MTLComputeCommandEncoder> encoder=[command computeCommandEncoder];
            if(!command || !encoder) throw std::runtime_error("Metal command creation failed");
            [encoder setComputePipelineState:p->pipeline];
            [encoder setBuffer:p->a offset:0 atIndex:0];[encoder setBuffer:p->v offset:0 atIndex:1];
            [encoder setBuffer:p->w offset:0 atIndex:2];[encoder setBuffer:p->status offset:0 atIndex:3];
            [encoder setBuffer:p->sweeps offset:0 atIndex:4];[encoder setBuffer:p->residual offset:0 atIndex:5];
            [encoder setBytes:&cfg length:sizeof(cfg) atIndex:6];
            [encoder dispatchThreadgroups:MTLSizeMake(count,1,1) threadsPerThreadgroup:MTLSizeMake(256,1,1)];
            [encoder endEncoding];[command commit];[command waitUntilCompleted];
            if(command.status!=MTLCommandBufferStatusCompleted)
                throw std::runtime_error(command.error ? command.error.localizedDescription.UTF8String : "Metal command did not complete");
            memcpy(vectors,p->v.contents,bytes);memcpy(values,p->w.contents,size_t(p->n)*count*4);
            memcpy(status,p->status.contents,count*4);memcpy(sweeps,p->sweeps.contents,count*4);
            memcpy(residual,p->residual.contents,count*4);
            seconds[0]=std::chrono::duration<double>(std::chrono::steady_clock::now()-started).count();
            if(command.GPUStartTime>0 && command.GPUEndTime>=command.GPUStartTime)
                seconds[1]=command.GPUEndTime-command.GPUStartTime;
            return 0;
        } catch(const std::exception& e) {error_text(error,size,e.what());return 1;}
    }
}
