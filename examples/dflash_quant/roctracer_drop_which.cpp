// Which kernels get dropped? One graph of K *distinct* kernels (knode<0..K-1>),
// replayed `iters` times. We parse the node index out of each recorded kernel
// name to build a per-node-position drop histogram, and report how records are
// grouped (distinct correlation ids / queues).
//
// Build: hipcc -O2 roctracer_drop_which.cpp -o which \
//          -I/opt/rocm/include -I/opt/rocm/include/roctracer \
//          -L/opt/rocm/lib -lroctracer64
// Run:   ./which [iters=200]   (K is fixed at 64 by template instantiation)

#include <unistd.h>
#include <hip/hip_runtime.h>
#include <roctracer.h>
#include <roctracer_ext.h>
#include <roctracer_hip.h>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <set>
#include <vector>

#define K 64

static std::mutex g_mu;
static long g_count[K] = {0};          // per-node-position recorded count
static long g_total = 0;               // total dispatch records
static std::set<unsigned long> g_corr; // distinct correlation ids
static std::set<unsigned long> g_queue;// distinct queue ids

// Distinct kernel per node position so the name encodes the index.
template <int ID>
__global__ void knode(float* x) { x[threadIdx.x] += (float)ID; }

template <int I>
struct Launcher {
  static void go(float* x, hipStream_t st) {
    Launcher<I - 1>::go(x, st);
    knode<I - 1><<<1, 256, 0, st>>>(x);  // launches knode<0>,<1>,...,<K-1>
  }
};
template <>
struct Launcher<0> {
  static void go(float*, hipStream_t) {}
};

// Parse the integer node id out of a (possibly mangled) symbol containing
// "knode". Mangled looks like _Z5knodeILi12EEvPf -> first digit run after
// "knode" is the id; demangled "knode<12>" works too.
static int parse_id(const char* name) {
  const char* p = strstr(name, "knode");
  if (!p) return -1;
  p += 5;
  while (*p && !isdigit((unsigned char)*p)) ++p;
  if (!*p) return -1;
  return atoi(p);
}

static void activity_cb(const char* begin, const char* end, void*) {
  std::lock_guard<std::mutex> lk(g_mu);
  auto* r = (const roctracer_record_t*)begin;
  auto* e = (const roctracer_record_t*)end;
  while (r < e) {
    if (r->op == HIP_OP_ID_DISPATCH && r->kernel_name) {
      g_total++;
      g_corr.insert((unsigned long)r->correlation_id);
      g_queue.insert((unsigned long)r->queue_id);
      int id = parse_id(r->kernel_name);
      if (id >= 0 && id < K) g_count[id]++;
    }
    roctracer_next_record(r, &r);
  }
}
static void api_cb(uint32_t, uint32_t, const void*, void*) {}

int main(int argc, char** argv) {
  int iters = argc > 1 ? atoi(argv[1]) : 200;

  float* buf;
  hipMalloc(&buf, 256 * sizeof(float));
  hipStream_t st;
  hipStreamCreate(&st);

  roctracer_set_properties(ACTIVITY_DOMAIN_HIP_API, nullptr);
  roctracer_enable_domain_callback(ACTIVITY_DOMAIN_HIP_API, api_cb, nullptr);
  roctracer_properties_t props;
  memset(&props, 0, sizeof(props));
  props.buffer_size = 0x100000;  // 1 MB, plenty
  props.buffer_callback_fun = activity_cb;
  roctracer_pool_t* pool = nullptr;
  roctracer_open_pool_expl(&props, &pool);
  roctracer_enable_domain_activity_expl(ACTIVITY_DOMAIN_HCC_OPS, pool);
  roctracer_start();

  hipStreamBeginCapture(st, hipStreamCaptureModeThreadLocal);
  Launcher<K>::go(buf, st);
  hipGraph_t graph;
  hipStreamEndCapture(st, &graph);
  hipGraphExec_t exec;
  hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0);
  for (int i = 0; i < iters; ++i) hipGraphLaunch(exec, st);
  hipStreamSynchronize(st);

  hipDeviceSynchronize();
  for (int t = 0; t < 50; ++t) {
    roctracer_flush_activity_expl(pool);
    usleep(1000);
  }
  roctracer_stop();

  long launched = (long)iters * K;
  printf("iters=%d  K=%d  launched=%ld  recorded=%ld  (%.1f%%)  dropped=%ld\n",
         iters, K, launched, g_total, 100.0 * g_total / launched,
         launched - g_total);
  printf("distinct correlation ids=%zu   distinct queue ids=%zu\n", g_corr.size(),
         g_queue.size());
  printf("\nper-node-position recorded / %d replays  (node : recorded  drop%%):\n",
         iters);
  for (int i = 0; i < K; ++i) {
    double dp = 100.0 * (iters - g_count[i]) / iters;
    printf("  %2d: %4ld  %5.1f%%%s", i, g_count[i], dp,
           ((i % 4) == 3) ? "\n" : "   ");
  }
  printf("\n");
  return 0;
}
