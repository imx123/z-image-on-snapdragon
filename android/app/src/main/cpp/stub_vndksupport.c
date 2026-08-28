/*
 * Minimal libvndksupport stub for zimage app.
 *
 * libOpenCL_adreno.so needs exactly one symbol from libvndksupport:
 *   android_load_sphal_library() -- loads a vendor (SPHAL) shared library
 *   in the vendor namespace. In a non-root app we cannot dlopen vendor
 *   libs anyway; returning NULL (failure) is safe: OpenCL core compute
 *   (clGetPlatformIDs / buffers / kernels) does not require gralloc/EGL
 *   interop, which is the only consumer of SPHAL here.
 */
#include <stddef.h>

__attribute__((visibility("default")))
void* android_load_sphal_library(const char* name) {
    (void)name;
    return NULL;
}
