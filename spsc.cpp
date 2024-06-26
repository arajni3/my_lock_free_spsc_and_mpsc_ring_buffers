#include "ring_buf.hpp"

template<typename DataType, unsigned length, unsigned version_granularity>
RingBuf<DataType, length, version_granularity>::RingBuf() {
  write_offset = 0;
  for (unsigned i = 0; i < version_granularity; ++i) {
    version_numbers[i].number.store(0, std::memory_order_relaxed);
  }
}

template<typename DataType, unsigned length, unsigned version_granularity>
void RingBuf<DataType, length, version_granularity>::write(DataType* data) {
  const unsigned version_idx = write_offset & (version_granularity - 1);
  std::atomic<std::size_t>& version_number = version_numbers[version_idx].number;


  /* Need release semantics for the second version number store to synchronize memcpy with it 
  (not needed for write offset). Need an explicit dependency of the memcpy on the first 
  version number store to synchronize it with the latter; otherwise, the latter can be 
  done with relaxed semantics.
  */

  volatile std::size_t write_guard = version_number.fetch_add(1, std::memory_order_relaxed);

  if (write_guard) { std::memcpy(&buf[write_offset], data, sizeof(DataType)); }
  write_offset = (write_offset + 1) & (length - 1);
  
  version_number.fetch_add(1, std::memory_order_release);
}

template<typename DataType, unsigned length, unsigned version_granularity>
unsigned RingBuf<DataType, length, version_granularity>::read(unsigned read_offset, DataType* ret_data) {
  const unsigned version_idx = read_offset & (version_granularity - 1);
  std::atomic<std::size_t>& version_number = version_numbers[version_idx].number;

  // need acquire semantics to synchronize with the memcpy (do first then check)
  do {
    std::memcpy(ret_data, &buf[read_offset], sizeof(DataType));
  } while (version_number.load(std::memory_order_acquire) & 1);
  return read_offset;
}