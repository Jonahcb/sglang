// Minimal repro: does roctracer deliver an async record for EVERY kernel
// replayed via hipGraphLaunch? One kernel, one graph.
//
// Build: hipcc -O2 roctracer_drop_min.cpp -o min \
//          -I/opt/rocm/include -I/opt/rocm/include/roctracer \
//          -L/opt/rocm/lib -lroctracer64
// Run:   ./min [iters=60] [kernels_per_graph=64]

#include <unistd.h>
#include <hip/hip_runtime.h>
#include <roctracer.h>
#include <roctracer_ext.h>
#include <roctracer_hip.h>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>

static std::atomic<long> g_recorded{0};

__global__ void k(float* x) { x[threadIdx.x] += 1.0f; }

// roctracer hands back packed activity records; count the kernel dispatches.
static void activity_cb(const char* begin, const char* end, void*) {
  auto* r = (const roctracer_record_t*)begin;
  auto* e = (const roctracer_record_t*)end;
  while (r < e) {
    if (r->op == HIP_OP_ID_DISPATCH && r->kernel_name) g_recorded++;
    roctracer_next_record(r, &r);
  }
}
static void api_cb(uint32_t, uint32_t, const void*, void*) {}  // must be armed

int main(int argc, char** argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 60;
  int K = argc > 2 ? atoi(argv[2]) : 64;
  size_t bufsz = argc > 3 ? (size_t)atol(argv[3]) : 0x4000;  // pool buffer bytes

  float* buf;
  hipMalloc(&buf, 256 * sizeof(float));
  hipStream_t st;
  hipStreamCreate(&st);

  // roctracer setup (same shape as kineto: API callback + 16KB HCC_OPS pool)
  roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, nullptr);
  roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, api_cb, nullptr);
  roctracer_properties_t props;
  memset(&props, 0, sizeof(props));
  props.buffer_size = bufsz;
  props.buffer_callback_fun = activity_cb;
  roctracer_pool_t* pool = nullptr;
  roctracer_open_pool_expl(&props, &pool);
  roctracer_enable_domain_activity_expl(ACTIVITY_DOMAIN_HCC_OPS, pool);
  roctracer_start();

  // Capture one graph with K kernels, then replay it `iters` times.
  hipStreamBeginCapture(st, hipStreamCaptureModeThreadLocal);
  for (int i = 0; i < K; ++i) k<<<1, 256, 0, st>>>(buf);
  hipGraph_t graph;
  hipStreamEndCapture(st, &graph);
  hipGraphExec_t exec;
  hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
  for (int i = 0; i < iters; ++i) hipGraphLaunch(exec, st);
  hipStreamSynchronize(st);

  // Drain everything (sync + repeated flush), like kineto's stop path.
  hipDeviceSynchronize();
  for (int t = 0; t < 50; ++t) {
    roctracer_flush_activity_expl(pool);
    usleep(1000);
  }
  roctracer_stop();

  long launched = (long)iters * K, rec = g_recorded.load();
  printf("bufsize %8zu B: launched %ld, recorded %ld  (%.1f%%)  -> %ld dropped\n",
         bufsz, launched, rec, 100.0 * rec / launched, launched - rec);
  return 0;
}
