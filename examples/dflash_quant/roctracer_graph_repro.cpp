// Standalone reproduction: does roctracer's async (HCC_OPS) stream deliver a
#include <unistd.h>
// kernel-dispatch record for EVERY kernel replayed via hipGraphLaunch?
//
// Mimics kineto's RoctracerLogger setup exactly:
//   - 16 KB (0x4000) HCC pool, buffer_callback_fun
//   - roctracer_enable_domain_activity_expl(ACTIVITY_DOMAIN_HCC_OPS, pool)
//   - on stop: hipDeviceSynchronize() + roctracer_flush_activity_expl(pool)
// (see third_party/kineto/libkineto/src/RoctracerLogger.cpp:282-416)
//
// Scenario mirrors SGLang spec-decode: each "iteration" launches a SMALL fast
// graph (draft, kdraft kernels) then a BIG slow graph (target, ktarget kernels).
// We count how many kernel-dispatch activity records roctracer actually hands
// back, per kernel name, vs how many were truly launched.
//
// Build: hipcc -O2 roctracer_graph_repro.cpp -o repro \
//          -I/opt/rocm/include -I/opt/rocm/include/roctracer \
//          -L/opt/rocm/lib -lroctracer64 -lroctx64
// Run:   ./repro [iters=60] [draft_kernels=8] [target_kernels=64] [mode=graph|eager]

#include <hip/hip_runtime.h>
#include <roctracer.h>
#include <roctracer_ext.h>
#include <roctracer_hip.h>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <vector>

#define HIP_CHECK(x)                                                       \
  do {                                                                     \
    hipError_t e = (x);                                                    \
    if (e != hipSuccess) {                                                 \
      fprintf(stderr, "HIP error %s at %s:%d\n", hipGetErrorString(e),     \
              __FILE__, __LINE__);                                         \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

#define RT_CHECK(x)                                                        \
  do {                                                                     \
    roctracer_status_t s = (x);                                            \
    if (s != ROCTRACER_STATUS_SUCCESS) {                                   \
      fprintf(stderr, "roctracer error %d (%s) at %s:%d\n", s,             \
              roctracer_error_string(), __FILE__, __LINE__);               \
      std::exit(1);                                                        \
    }                                                                      \
  } while (0)

// Two distinct kernels so we can tell which graph each record came from,
// exactly like F8BS-in-draft vs target kernels in the real trace.
__global__ void kdraft(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = x[i] * 1.0001f + 1.0f;
}
__global__ void ktarget(float* x, int n) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) x[i] = x[i] * 0.9999f - 0.5f;
}

// ---- roctracer collection state ----
static std::mutex g_mu;
static std::atomic<long> g_draft_records{0};
static std::atomic<long> g_target_records{0};
static std::atomic<long> g_other_dispatch{0};
// ordered log of (kernel_name_tag) as delivered, to inspect "recovered at flush"
static std::vector<char> g_order;  // 'd' = draft, 't' = target

// kineto enables the HIP_API domain callback too (RoctracerLogger.cpp:347).
// On this ROCm, async HCC_OPS records are only emitted for API calls that have
// an active callback, so we must register one even though we ignore the data.
static void api_cb(uint32_t, uint32_t, const void*, void*) {}

static std::atomic<long> g_total_records{0};
static int g_dbg = 0;  // dump first N records when DEBUG_RECORDS set

static void activity_cb(const char* begin, const char* end, void* /*arg*/) {
  std::lock_guard<std::mutex> lk(g_mu);
  const roctracer_record_t* r = (const roctracer_record_t*)begin;
  const roctracer_record_t* e = (const roctracer_record_t*)end;
  while (r < e) {
    g_total_records++;
    if (g_dbg > 0) {
      fprintf(stderr,
              "  rec domain=%u op=%u kind=%u corr=%lu name=%s\n", r->domain,
              r->op, r->kind, (unsigned long)r->correlation_id,
              r->kernel_name ? r->kernel_name : "(null)");
      g_dbg--;
    }
    // kineto's filter: dispatch op, not a barrier, has a kernel name.
    if (r->op == HIP_OP_ID_DISPATCH && r->kernel_name != nullptr) {
      std::string nm(r->kernel_name);
      if (nm.find("kdraft") != std::string::npos) {
        g_draft_records++;
        g_order.push_back('d');
      } else if (nm.find("ktarget") != std::string::npos) {
        g_target_records++;
        g_order.push_back('t');
      } else {
        g_other_dispatch++;
      }
    }
    RT_CHECK(roctracer_next_record(r, &r));
  }
}

// Capture `count` launches of `which` kernel into a graph via stream capture.
static hipGraphExec_t build_graph(hipStream_t st, int which, int count,
                                  float* buf, int n) {
  int block = 256, grid = (n + block - 1) / block;
  HIP_CHECK(hipStreamBeginCapture(st, hipStreamCaptureModeThreadLocal));
  for (int k = 0; k < count; ++k) {
    if (which == 0)
      kdraft<<<grid, block, 0, st>>>(buf, n);
    else
      ktarget<<<grid, block, 0, st>>>(buf, n);
  }
  hipGraph_t graph;
  HIP_CHECK(hipStreamEndCapture(st, &graph));
  hipGraphExec_t exec;
  HIP_CHECK(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
  HIP_CHECK(hipGraphDestroy(graph));
  return exec;
}

int main(int argc, char** argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 60;
  int draftK = argc > 2 ? atoi(argv[2]) : 8;
  int targetK = argc > 3 ? atoi(argv[3]) : 64;
  std::string mode = argc > 4 ? argv[4] : "graph";
  if (getenv("DEBUG_RECORDS")) g_dbg = atoi(getenv("DEBUG_RECORDS"));

  const int N = 1 << 16;
  float* buf;
  HIP_CHECK(hipMalloc(&buf, N * sizeof(float)));
  HIP_CHECK(hipMemset(buf, 0, N * sizeof(float)));

  hipStream_t st;
  HIP_CHECK(hipStreamCreate(&st));

  // ---- roctracer setup, identical shape to kineto RoctracerLogger ----
  RT_CHECK(roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, nullptr));
  RT_CHECK(roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, api_cb,
                                            nullptr));
  roctracer_properties_t props;
  memset(&props, 0, sizeof(props));
  props.buffer_size = 0x4000;  // 16 KB, same as kineto
  props.buffer_callback_fun = activity_cb;
  roctracer_pool_t* pool = nullptr;
  RT_CHECK(roctracer_open_pool_expl(&props, &pool));
  RT_CHECK(roctracer_enable_domain_activity_expl(ACTIVITY_DOMAIN_HCC_OPS, pool));
  roctracer_start();

  int block = 256, grid = (N + block - 1) / block;

  if (mode == "graph") {
    hipGraphExec_t gd = build_graph(st, 0, draftK, buf, N);
    hipGraphExec_t gt = build_graph(st, 1, targetK, buf, N);
    for (int i = 0; i < iters; ++i) {
      HIP_CHECK(hipGraphLaunch(gd, st));  // draft (small/fast)
      HIP_CHECK(hipGraphLaunch(gt, st));  // target (big/slow)
    }
    HIP_CHECK(hipStreamSynchronize(st));
  } else {  // eager control: identical kernels, no graph
    for (int i = 0; i < iters; ++i) {
      for (int k = 0; k < draftK; ++k) kdraft<<<grid, block, 0, st>>>(buf, N);
      for (int k = 0; k < targetK; ++k) ktarget<<<grid, block, 0, st>>>(buf, N);
    }
    HIP_CHECK(hipStreamSynchronize(st));
  }

  // ---- stop exactly like kineto: sync, then a flush loop, then stop
  // (RoctracerLogger.cpp:394-415 polls flush up to 50x with 1ms sleeps) ----
  HIP_CHECK(hipDeviceSynchronize());
  for (int t = 0; t < 50; ++t) {
    RT_CHECK(roctracer_flush_activity_expl(pool));
    usleep(1000);
  }
  roctracer_stop();

  long dr = g_draft_records.load(), tr = g_target_records.load();
  long exp_d = (long)iters * draftK, exp_t = (long)iters * targetK;
  printf("\n=== roctracer HCC_OPS kernel-dispatch records (%s mode) ===\n",
         mode.c_str());
  printf("iterations: %d   draft kernels/iter: %d   target kernels/iter: %d\n",
         iters, draftK, targetK);
  printf("DRAFT  (small/fast graph): recorded %5ld / %5ld launched  (%.1f%%)\n",
         dr, exp_d, 100.0 * dr / exp_d);
  printf("TARGET (big/slow graph):   recorded %5ld / %5ld launched  (%.1f%%)\n",
         tr, exp_t, 100.0 * tr / exp_t);
  printf("other dispatch records: %ld    total records of any kind: %ld\n",
         g_other_dispatch.load(), g_total_records.load());

  // Where in the delivery order do the draft records land? (recovered-at-flush?)
  if (!g_order.empty()) {
    int n = g_order.size();
    int firstD = -1, lastD = -1, cntD = 0;
    for (int i = 0; i < n; ++i)
      if (g_order[i] == 'd') {
        if (firstD < 0) firstD = i;
        lastD = i;
        cntD++;
      }
    printf(
        "delivery order: %d total records; draft records at positions "
        "[%d .. %d] of %d\n",
        n, firstD, lastD, n);
    // decile histogram of draft records across delivery order
    int dec[10] = {0};
    for (int i = 0; i < n; ++i)
      if (g_order[i] == 'd') dec[(int)((long)i * 10 / n)]++;
    printf("draft records per delivery-order decile [start..end]: ");
    for (int i = 0; i < 10; ++i) printf("%d ", dec[i]);
    printf("\n");
  }
  return 0;
}
