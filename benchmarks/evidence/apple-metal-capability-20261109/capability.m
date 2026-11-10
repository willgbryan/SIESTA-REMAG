#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
int main(void) {
  @autoreleasepool {
    id<MTLDevice> device = MTLCreateSystemDefaultDevice();
    if (!device) { puts("No Metal device visible"); return 2; }
    NSMutableDictionary *report = [@{@"device":device.name,
      @"unified_memory":@(device.hasUnifiedMemory),
      @"recommended_working_set_bytes":@(device.recommendedMaxWorkingSetSize),
      @"max_buffer_bytes":@(device.maxBufferLength)} mutableCopy];
    for (NSString *type in @[@"float", @"double"]) {
      NSString *source = [NSString stringWithFormat:
        @"#include <metal_stdlib>\nusing namespace metal;\nkernel void probe(device %@* a [[buffer(0)]], uint i [[thread_position_in_grid]]) { a[i] = a[i] + (%@)1; }",type,type];
      NSError *error = nil;
      id<MTLLibrary> library = [device newLibraryWithSource:source options:nil error:&error];
      report[type] = @{@"compiled":@(library != nil), @"diagnostic":error ? error.localizedDescription : @""};
    }
    NSData *json = [NSJSONSerialization dataWithJSONObject:report options:NSJSONWritingPrettyPrinted error:nil];
    puts([[NSString alloc] initWithData:json encoding:NSUTF8StringEncoding].UTF8String);
  }
  return 0;
}
