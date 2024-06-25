#include <atomic>
#include <cstring>
#define ALIGN_NO_FALSE_SHARING (64 * 2) // align to two cache lines because of prefetching

/* Lock-free ring buffer with SPSC and MPSC implementations. Typically only a single 
consumer exists. The writer is in fact wait-free in the SPSC case. The length and version 
granularity must be powers of 2 to make modulo as fast as possible, and version_granularity
must divide length (i.e., be <= length). Ideally, DataType should be aligned to avoid 
false sharing.
*/
template<typename DataType, unsigned length, unsigned version_granularity = length>
struct RingBuf {
  union {
    alignas(ALIGN_NO_FALSE_SHARING) std::atomic<unsigned> atomic_global_write_offset; // for MPSC
    alignas(ALIGN_NO_FALSE_SHARING) unsigned write_offset; // for SPSC
  };

  struct alignas(ALIGN_NO_FALSE_SHARING) __version_alignment_wrapper {
    std::atomic<std::size_t> number;
  }
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

  // underlying buffer
  DataType buf[length];
  
  void write(DataType* data);

  /* Returns the new read offset. For a single consumer, 
  the reader will trivially start at 0 and will increment its read offset after each 
  successful read by setting it to the output of this function. Returning the new offset 
  simply copies less bytes (4 versus 8 on 64-bit systems), so it is more efficient.
  */
  unsigned read(unsigned read_offset, DataType* ret_data);

  RingBuf();
};