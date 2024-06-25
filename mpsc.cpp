#include <atomic>
#include <cstring>
#include "ring_buf.hpp"

template<typename DataType, unsigned length, unsigned version_granularity>
RingBuf<DataType, length, version_granularity>::RingBuf() {
  atomic_global_write_offset.store(0, std::memory_order_relaxed);
  for (unsigned i = 0; i < version_granularity; ++i) {
    version_numbers[i].store(0, std::memory_order_relaxed);
  }
}

template<typename DataType, unsigned length, unsigned version_granularity>
void RingBuf<DataType, length, version_granularity>::write(DataType* data) {
  unsigned local_offset;
  unsigned version_idx;
  std::atomic<std::size_t>* version_number = nullptr;

  /* Description: We want to try to write to the current offset if there is no 
  contention with another writer, i.e., if the global write offset ends up being the same 
  as the loaded local offset. 


    (1) If this is the case, the global write offset will be 
  incremented by 1 for the next writer. We do that with release semantics so that the 
  store is synchronized with the prior load of the same atomic (which can be 
  relaxed because it cannot occur after the release operation). The later memcpy 
  is already synchronized because it explicitly depends on the local offset value.

    (2) Otherwise, the global write offset will in fact 
  be greater than the loaded local offset, and we have to try again. We fail with relaxed 
  semantics because no store happened, hence no synchronization is needed with the 
  prior load of the same atomic. The later memcpy is already synchronized because it 
  explicitly depends on the local offset value, which will be recomputed.

    

  The version number increment is done (possibly repeatedly) in the CAS loop. This 
  is so that sequential consistency is not needed to synchronize the store of the 
  global write offset with the stores of the version number. Retry is detected by setting 
  the initial value of the version number pointer to null, in which case it will be 
  non-null only upon retry, in which case the version number must be subtracted. This 
  will correctly synchronize with the reader because the reader will detect contention 
  with a writer if and only if the version number is positive. The subtraction in the 
  loop can be done with relaxed semantics because there is nothing to synchronize with 
  if contention was detected, and if the new loop iteration yields the same version 
  number, then the later addition in the same loop iteration, which is done with 
  release semantics, will synchronize with it anyway; this release operation will then 
  also synchronize with the later memcpy, and likewise so will the final subtraction, 
  which has release semantics.
  */

  do {
    if (version_number) { version_number->fetch_sub(1, std::memory_order_relaxed); }
    local_offset = atomic_global_write_offset.load(std::memory_order_relaxed);
    version_idx = local_offset & (version_granularity - 1);
    version_number->fetch_add(1, std::memory_order_release);
  } while (!atomic_global_write_offset.compare_exchange_weak(
    local_offset, 
    local_offset + 1, 
    std::memory_order_release,
    std::memory_order_relaxed
  ));

  std::memcpy(&buf[local_offset], data, sizeof(DataType));

  version_number->fetch_sub(1, std::memory_order_release);
}

template<typename DataType, unsigned length, unsigned version_granularity>
unsigned RingBuf<DataType, length, version_granularity>::read(unsigned read_offset, DataType* ret_data) {
  const unsigned version_idx = read_offset & (version_granularity - 1);
  std::atomic<std::size_t>& version_number = version_numbers[version_idx];

  // need acquire semantics to synchronize with the memcpy (do first then check)
  do {
    std::memcpy(ret_data, &buf[read_offset], sizeof(DataType));
  } while (version_number.load(std::memory_order_acquire));
  return read_offset;
}