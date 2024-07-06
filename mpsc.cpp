#include "ring_buf.hpp"

template<typename DataType, unsigned length, unsigned version_granularity>
RingBuf<DataType, length, version_granularity>::RingBuf() {
  prod_u.atomic_global_write_sequence_number.store(0, std::memory_order_relaxed);
  read_sequence_number = 0;
  for (unsigned i = 0; i < version_granularity; ++i) {
    version_numbers[i].number.store(0, std::memory_order_relaxed);
  }
  versioned_DataType start{ .sequence_number = 0 };
  std::fill(buf, buf + length, start);
}

template<typename DataType, unsigned length, unsigned version_granularity>
void RingBuf<DataType, length, version_granularity>::write(DataType* data) {
  uint64_t local_sequence_number;
  unsigned version_idx;
  std::atomic<uint64_t>* version_number_ptr = nullptr;
  volatile uint64_t write_guard = 0;

  /* Description: We want to try to write to the current sequence number if there is no 
  contention with another writer, i.e., if the global write sequence number ends up being the same 
  as the loaded local sequence number. 


    (1) If this is the case, the global write sequence number will be 
  incremented by 1 for the next writer. We do that with release semantics so that the 
  store is synchronized with the prior load of the same atomic (which can be 
  relaxed because it cannot occur after the release operation). The later memcpy 
  is already synchronized because it explicitly depends on the local sequence number value.

    (2) Otherwise, the global write sequence number will in fact 
  be greater than the loaded local sequence number, and we have to try again. We fail with relaxed 
  semantics because no store happened, hence no synchronization is needed with the 
  prior load of the same atomic. The later memcpy is already synchronized because it 
  explicitly depends on the local sequence number value, which will be recomputed. The atomic 
  sequence number in the loop iteration can be done with relaxed semantics because the CAS 
  operation in the branch explicitly depends on the local sequence number hence synchronizes with the very 
  first iteration and trivially synchronizes with future iterations because future iterations fall 
  under the branch.

    

  The version number increment is done (possibly repeatedly) in the CAS loop. This 
  is so that sequential consistency is not needed to synchronize the store of the 
  global write sequence number with the stores of the version number. 
  
  Retry is detected by setting the initial value of the version number pointer to null, 
  in which case it will be non-null only upon retry, in which case the version number 
  must be compared with the new version number pointer, and if they are not the same, then 
  the old version number must be corrected (decremented by 1) and the new version number should be 
  claimed (incremented by 1). 
  
  This will correctly synchronize with the reader because the reader will detect contention
  with a writer if and only if the version number is positive, and the increment is done with 
  relaxed semantics for efficiency but the producer memcpy is protected by an explicit, volatile 
  dependency on said increment hence synchronizes with the increment. 
  
  If the new version number location is different, then a subsequent subtraction of the old version 
  number is required; this subtraction can be done with relaxed semantics because the CAS branch 
  synchronizes it with the addition that took place in the previous loop iteration due to the subsequent 
  local sequence number load being dependent on the CAS branch as described above. If the new loop iteration 
  yields the same version number, then there is no change in the value of the version number on this loop 
  iteration, and on the very first loop iteration, the only version number operation is the single 
  (relaxed) increment, which is thread-safe as explained above.
  */

  do {
    local_sequence_number = prod_u.atomic_global_write_sequence_number.load(std::memory_order_relaxed);
    version_idx = local_sequence_number & (version_granularity - 1);
    std::atomic<uint64_t>* next_version_number_ptr = &version_numbers[version_idx].number;

    if (next_version_number_ptr != version_number_ptr) {
      if (version_number_ptr) { version_number_ptr->fetch_sub(1, std::memory_order_relaxed); }
      write_guard = next_version_number_ptr->fetch_add(1, std::memory_order_relaxed);
      version_number_ptr = next_version_number_ptr;
    }
  } while (!prod_u.atomic_global_write_sequence_number.compare_exchange_weak(
    local_sequence_number, 
    local_sequence_number + 1, 
    std::memory_order_release,
    std::memory_order_relaxed
  ));

  versioned_DataType entry{*data, local_sequence_number + 1}; // first written sequence number is 1
  if (write_guard) { std::memcpy(&buf[local_sequence_number & (length - 1)], &entry, sizeof(versioned_DataType)); }

  version_number_ptr->fetch_sub(1, std::memory_order_release);
}

template<typename DataType, unsigned length, unsigned version_granularity>
bool RingBuf<DataType, length, version_granularity>::read(DataType* ret_data) {
  const unsigned version_idx = read_sequence_number & (version_granularity - 1);
  std::atomic<uint64_t>& version_number = version_numbers[version_idx].number;

  // need acquire semantics to synchronize with the memcpy (do first then check)
  versioned_DataType entry;
  do {
    std::memcpy(&entry, &buf[read_sequence_number & (length - 1)], sizeof(versioned_DataType));
  } while (version_number.load(std::memory_order_acquire));
  
  std::memcpy(ret_data, &entry.data, sizeof(DataType)); // copy to client regardless of success to avoid branch predictions
  unsigned char success = (uint64_t)(read_sequence_number - entry.sequence_number) >> 63; // success iff sequence number > read sequence number
  read_sequence_number += success;
  return success;
}