#include "ring_buf.hpp"

template<typename DataType, unsigned length, unsigned version_granularity>
RingBuf<DataType, length, version_granularity>::RingBuf() {
  prod_u.write_sequence_number = 0;
  read_sequence_number = 0;
  for (unsigned i = 0; i < version_granularity; ++i) {
    version_numbers[i].number.store(0, std::memory_order_relaxed);
  }
  versioned_DataType start{ .sequence_number = 0 };
  std::fill(buf, buf + length, start);
}

template<typename DataType, unsigned length, unsigned version_granularity>
void RingBuf<DataType, length, version_granularity>::write(DataType* data) {
  const unsigned version_idx = prod_u.write_sequence_number & (version_granularity - 1);
  std::atomic<uint64_t>& version_number = version_numbers[version_idx].number;


  /* Need release semantics for the second version number store to synchronize memcpy with it.
  Need an explicit dependency of the memcpy on the first version number store to synchronize it 
  with the latter; otherwise, the latter can be done with relaxed semantics.
  */

  volatile uint64_t write_guard = version_number.fetch_add(1, std::memory_order_relaxed);


  const uint64_t write_sequence_number = prod_u.write_sequence_number;
  versioned_DataType entry{*data, ++prod_u.write_sequence_number}; // first written sequence number is 1
  if (write_guard) { std::memcpy(&buf[write_sequence_number & (length - 1)], &entry, sizeof(versioned_DataType)); }
  
  version_number.fetch_add(1, std::memory_order_release);
}

template<typename DataType, unsigned length, unsigned version_granularity>
bool RingBuf<DataType, length, version_granularity>::read(DataType* ret_data) {
  const unsigned version_idx = read_sequence_number & (version_granularity - 1);
  std::atomic<uint64_t>& version_number = version_numbers[version_idx].number;

  versioned_DataType entry;
  // need store and load fences to synchronize check with the memcpy (do first then check; memcpy before store fence and check after load fence)
  do {
    std::memcpy(&entry, &buf[read_sequence_number & (length - 1)], sizeof(versioned_DataType));
    std::atomic_thread_fence(std::memory_order_release);
    std::atomic_thread_fence(std::memory_order_acquire);
  } while (version_number.load(std::memory_order_relaxed) & 1);

  unsigned char success = (uint64_t)(read_sequence_number - entry.sequence_number) >> 63; // success iff sequence number > read sequence number
  if (success) { std::memcpy(ret_data, &entry.data, sizeof(DataType)); } // conditional since DataType may be large, e.g., a whole network packet
  read_sequence_number += success;
  return success;
}