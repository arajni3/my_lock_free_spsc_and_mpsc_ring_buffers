#pragma once
#include <atomic>
#include <cstring>
#include <algorithm>
#include <type_traits>
#include <numeric>
#include <cstdint>
#define ALIGN_NO_FALSE_SHARING (64 * 2) // align to two cache lines because of prefetching

/* Lock-free ring buffer with SPSC and MPSC implementations. Typically only a single 
consumer exists. The writer is in fact wait-free in the SPSC case. The length and version 
granularity must be powers of 2 to make modulo as fast as possible, and version_granularity
must divide length (i.e., be <= length). Finally, DataType should be a POD struct.
*/
template<typename DataType, unsigned length, unsigned version_granularity = length>
struct RingBuf {
  static_assert(length && !(length & (length - 1)), "length must be a power of 2");
  static_assert(version_granularity && !(version_granularity & (version_granularity - 1)), "version granularity must be a power of 2");
  static_assert(!(length & (version_granularity - 1)), "version granularity must divide length");
  static_assert(std::is_trivially_copyable_v<DataType>, "DataType must be POD (to support memcpy)");

  union // the write offset member is not automatically wrapped because it also serves as a global sequence number to detect unwritten/stale entries
  {
    alignas(ALIGN_NO_FALSE_SHARING) std::atomic<uint64_t> atomic_global_write_offset; // for MPSC
    alignas(ALIGN_NO_FALSE_SHARING) uint64_t write_offset; // for SPSC
  } prod_u;

  struct alignas(ALIGN_NO_FALSE_SHARING) __version_alignment_wrapper {
    std::atomic<uint64_t> number;
  };
  /* Version numbers for the ring buffer.
  
  For SP, a version number is odd if its region is 
  currently being written to by a writer and even otherwise. Each version number starts 
  at 0 (even). A writer always adds to a version number, never subtract, to use the same 
  exact instruction for every store operation; this maximizes icache efficiency.

  For MP, a version number is 1 if its rewriter always adds to a version number to take control of it but subtracts when
  relinquishing control of the region to the reader. This is because a producer may have 
  incremented the version number on a stale append region, so it must later correct it. 
  Hence, in MP, a version number is actually a writer refcount.
  */
  __version_alignment_wrapper version_numbers[version_granularity];

  struct __unaligned_versioned_DataType {
    DataType data;
    uint64_t sequence_number;
  };
  static constexpr unsigned align_to_no_false_sharing() {
    unsigned gcd = ALIGN_NO_FALSE_SHARING;
    while (sizeof(__unaligned_versioned_DataType) % gcd && ALIGN_NO_FALSE_SHARING % gcd) { 
      gcd >>= 1; // < ALIGN_NO_FALSE_SHARING is always a power of 2
    } 

    unsigned lcm_alignment = (sizeof(__unaligned_versioned_DataType) / gcd) * ALIGN_NO_FALSE_SHARING; // divide by gcd first to avoid overflow
    unsigned final_alignment = 1;
    while (final_alignment < lcm_alignment) { final_alignment <<= 1; }
    return final_alignment;
  }
  struct alignas(align_to_no_false_sharing()) versioned_DataType {
    DataType data;
    uint64_t sequence_number;
  };
  // underlying buffer
  versioned_DataType buf[length];
  
  /* For writes, it is not expected that so many writes will occur without any reads 
  in-between that unread entries will be overwritten, so, for efficiency, overflow is not checked.
  */
  void write(DataType* data);

  /* Returns the new read sequence number (not wrapped). For a single consumer, 
  the reader will trivially start at 0 and will increment its read sequence number after each 
  successful read by setting it to the output of this function; it is not expected that so many 
  writes will occur without any reads in-between that unread entries will be overwritten, so, for 
  efficiency, overflow is not checked. If the current entry to read was not written or is stale (i.e., 
  if the entry's sequence number is not greater than the input sequence number (sequence numbers in 
  the ring buffer are 0 by default but written ones start at 1)), then the returned number is the same 
  as the input one.
  */
  uint64_t read(uint64_t read_offset, DataType* ret_data);

  RingBuf();
};