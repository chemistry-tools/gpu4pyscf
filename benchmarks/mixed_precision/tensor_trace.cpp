// Record CUDA 13 kernel names separately from benchmark timing.
#include <cupti.h>
#include <cstdlib>
#include <fstream>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

namespace {
std::mutex guard;
std::vector<std::pair<unsigned long long, std::string>> kernels;
size_t dropped = 0;
CUptiResult callback_error = CUPTI_SUCCESS;
bool registered = false;

void CUPTIAPI request(uint8_t **buffer, size_t *size, size_t *records) {
    *size = 1 << 20;
    *records = 0;
    if (posix_memalign(reinterpret_cast<void **>(buffer), 8, *size)) {
        *buffer = nullptr;
        *size = 0;
    }
}

void CUPTIAPI complete(CUcontext context, uint32_t stream, uint8_t *buffer,
                       size_t size, size_t valid) {
    std::lock_guard<std::mutex> lock(guard);
    CUpti_Activity *record = nullptr;
    CUptiResult status;
    while ((status = cuptiActivityGetNextRecord(buffer, valid, &record)) == CUPTI_SUCCESS) {
        if (record->kind == CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL) {
            auto *kernel = reinterpret_cast<CUpti_ActivityKernel10 *>(record);
            const char *name = kernel->name;
            kernels.emplace_back(kernel->end - kernel->start, name ? name : "");
        }
    }
    if (status != CUPTI_ERROR_MAX_LIMIT_REACHED) callback_error = status;
    size_t count = 0;
    status = cuptiActivityGetNumDroppedRecords(context, stream, &count);
    if (status != CUPTI_SUCCESS) callback_error = status;
    dropped += count;
    std::free(buffer);
}
}

extern "C" int trace_start() {
    kernels.clear();
    dropped = 0;
    callback_error = CUPTI_SUCCESS;
    if (!registered) {
        auto status = cuptiActivityRegisterCallbacks(request, complete);
        if (status != CUPTI_SUCCESS) return status;
        registered = true;
    }
    return cuptiActivityEnable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
}

extern "C" int trace_stop(const char *path) {
    auto status = cuptiActivityFlushAll(CUPTI_ACTIVITY_FLAG_FLUSH_FORCED);
    if (status != CUPTI_SUCCESS) return status;
    status = cuptiActivityDisable(CUPTI_ACTIVITY_KIND_CONCURRENT_KERNEL);
    if (status != CUPTI_SUCCESS) return status;
    std::lock_guard<std::mutex> lock(guard);
    std::ofstream output(path);
    output << "# dropped_records " << dropped << '\n';
    for (const auto &kernel : kernels) {
        output << kernel.first << '\t' << kernel.second << '\n';
    }
    if (!output) return -1;
    return callback_error;
}
