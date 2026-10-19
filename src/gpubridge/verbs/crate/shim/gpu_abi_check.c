/* Compile-only check of what src/gpu.rs takes from the GPU drivers' headers: every
 * constant it hard-codes and the signature of every function it loads with dlsym
 * (docs/VERBS_TRANSPORT.md, "GPU memory"). Not part of the library: gpu.rs loads the
 * drivers at run time, so nothing builds against these headers. Run one section at a
 * time against the real headers; it must compile without a diagnostic:
 *
 *   cc -fsyntax-only -Werror -DGBV_CHECK_CUDA -I<cuda>/include gpu_abi_check.c
 *   cc -fsyntax-only -Werror -DGBV_CHECK_HIP -D__HIP_PLATFORM_AMD__ -I<rocm>/include gpu_abi_check.c
 *   cc -fsyntax-only -Werror -DGBV_CHECK_HSA -I<dir with hsa.h, hsa_ext_amd.h> gpu_abi_check.c
 */
#include <stddef.h>
#include <stdint.h>

#define SAME(a, b) _Static_assert((a) == (b), #a " is " #b)

#ifdef GBV_CHECK_CUDA
#include <cuda.h>
SAME(CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_SUPPORTED, 116);
SAME(CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_FLUSH_WRITES_OPTIONS, 117);
SAME(CU_DEVICE_ATTRIBUTE_GPU_DIRECT_RDMA_WRITES_ORDERING, 118);
SAME(CU_DEVICE_ATTRIBUTE_DMA_BUF_SUPPORTED, 124);
SAME(CU_POINTER_ATTRIBUTE_MEMORY_TYPE, 2);
SAME(CU_POINTER_ATTRIBUTE_SYNC_MEMOPS, 6);
SAME(CU_POINTER_ATTRIBUTE_BUFFER_ID, 7);
SAME(CU_POINTER_ATTRIBUTE_IS_MANAGED, 8);
SAME(CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL, 9);
SAME(CU_POINTER_ATTRIBUTE_IS_GPU_DIRECT_RDMA_CAPABLE, 15);
SAME(CU_MEMORYTYPE_DEVICE, 2);
SAME(CU_MEM_RANGE_HANDLE_TYPE_DMA_BUF_FD, 1);
/* gpubridge.ucx.gdr's flush arguments, which the verbs bridge uses too. */
SAME(CU_FLUSH_GPU_DIRECT_RDMA_WRITES_OPTION_HOST, 1);
SAME(CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TARGET_CURRENT_CTX, 0);
SAME(CU_FLUSH_GPU_DIRECT_RDMA_WRITES_TO_OWNER, 100);
SAME(CU_GPU_DIRECT_RDMA_WRITES_ORDERING_OWNER, 100);
/* gpu.rs passes these as c_int, u64 and usize. */
SAME(sizeof(CUresult), 4);
SAME(sizeof(CUdevice), 4);
SAME(sizeof(CUdeviceptr), 8);
SAME(sizeof(CUdevice_attribute), 4);
SAME(sizeof(CUpointer_attribute), 4);
SAME(sizeof(CUmemRangeHandleType), 4);
/* The functions gpu.rs looks up, by the names it looks up (cuda.h maps the
 * unversioned ones to these with #define). */
CUresult (*const gbv_init)(unsigned int) = cuInit;
CUresult (*const gbv_version)(int *) = cuDriverGetVersion;
CUresult (*const gbv_error)(CUresult, const char **) = cuGetErrorString;
CUresult (*const gbv_device)(CUdevice *, int) = cuDeviceGet;
CUresult (*const gbv_attr)(int *, CUdevice_attribute, CUdevice) = cuDeviceGetAttribute;
CUresult (*const gbv_ptr_get)(void *, CUpointer_attribute, CUdeviceptr) = cuPointerGetAttribute;
CUresult (*const gbv_ptr_set)(const void *, CUpointer_attribute, CUdeviceptr) = cuPointerSetAttribute;
CUresult (*const gbv_range)(CUdeviceptr *, size_t *, CUdeviceptr) = cuMemGetAddressRange_v2;
CUresult (*const gbv_handle)(void *, CUdeviceptr, size_t, CUmemRangeHandleType,
                             unsigned long long) = cuMemGetHandleForAddressRange;
CUresult (*const gbv_retain)(CUcontext *, CUdevice) = cuDevicePrimaryCtxRetain;
CUresult (*const gbv_release)(CUdevice) = cuDevicePrimaryCtxRelease_v2;
CUresult (*const gbv_push)(CUcontext) = cuCtxPushCurrent_v2;
CUresult (*const gbv_pop)(CUcontext *) = cuCtxPopCurrent_v2;
CUresult (*const gbv_flush)(CUflushGPUDirectRDMAWritesTarget,
                            CUflushGPUDirectRDMAWritesScope) = cuFlushGPUDirectRDMAWrites;
#endif

#ifdef GBV_CHECK_HIP
#include <hip/hip_runtime_api.h>
SAME(HIP_POINTER_ATTRIBUTE_MEMORY_TYPE, 2);
SAME(HIP_POINTER_ATTRIBUTE_BUFFER_ID, 7);
SAME(HIP_POINTER_ATTRIBUTE_DEVICE_ORDINAL, 9);
SAME(hipMemoryTypeDevice, 2);
SAME(sizeof(hipError_t), 4);
SAME(sizeof(hipPointer_attribute), 4);
SAME(sizeof(hipDeviceptr_t), sizeof(void *));
hipError_t (*const gbv_hip_version)(int *) = hipRuntimeGetVersion;
const char *(*const gbv_hip_error)(hipError_t) = hipGetErrorString;
hipError_t (*const gbv_hip_ptr)(void *, hipPointer_attribute, hipDeviceptr_t) =
    hipPointerGetAttribute;
hipError_t (*const gbv_hip_range)(hipDeviceptr_t *, size_t *, hipDeviceptr_t) =
    hipMemGetAddressRange;
#endif

#ifdef GBV_CHECK_HSA
#include <hsa.h>
#include <hsa_ext_amd.h>
SAME(HSA_STATUS_SUCCESS, 0);
SAME(HSA_SYSTEM_INFO_VERSION_MAJOR, 0);
SAME(HSA_SYSTEM_INFO_VERSION_MINOR, 1);
SAME(HSA_AMD_SYSTEM_INFO_DMABUF_SUPPORTED, 0x204);
SAME(sizeof(hsa_status_t), 4);
SAME(sizeof(hsa_system_info_t), 4);
hsa_status_t (*const gbv_hsa_init)(void) = hsa_init;
hsa_status_t (*const gbv_hsa_string)(hsa_status_t, const char **) = hsa_status_string;
hsa_status_t (*const gbv_hsa_info)(hsa_system_info_t, void *) = hsa_system_get_info;
hsa_status_t (*const gbv_hsa_export)(const void *, size_t, int *, uint64_t *) =
    hsa_amd_portable_export_dmabuf;
hsa_status_t (*const gbv_hsa_close)(int) = hsa_amd_portable_close_dmabuf;
#endif
