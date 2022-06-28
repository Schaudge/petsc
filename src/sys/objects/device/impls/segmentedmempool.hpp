#ifndef PETSC_SEGMENTEDMEMPOOL_HPP
#define PETSC_SEGMENTEDMEMPOOL_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/cpp/register_finalize.hpp>
#include <petsc/private/cpputil.hpp>
#include <limits>
#include <deque>
#include <atomic>

namespace Petsc {

namespace Device {

namespace Impl {

template <typename AtomicFlagType>
struct MemoryChunk {
  using atomic_flag_type = AtomicFlagType;
  using size_type        = std::size_t;

  const size_type  start;
  const size_type  size;
  atomic_flag_type claimed = ATOMIC_FLAG_INIT;

  MemoryChunk(size_type start_, size_type size_) noexcept : start(start_), size(size_) { }

  explicit MemoryChunk(size_type size_) noexcept : MemoryChunk(0, size_) { }

  explicit MemoryChunk(const MemoryChunk<AtomicFlagType> &) noexcept : start(0), size(0) {
    SETERRABORT(PETSC_COMM_SELF, PETSC_ERR_PLIB, "This routine should never be called. Atomic flags are not copyable, this constructor exists purely to allow gcc's implementation of std::contruct() to compile");
  }

  void                 release(std::memory_order = std::memory_order_relaxed) noexcept;
  PETSC_NODISCARD bool try_claim(size_type) noexcept;
  PetscErrorCode       force_claim(std::memory_order = std::memory_order_seq_cst) noexcept;
};

/*
  MemoryChunk::release - release claim of a chunk

  Input Parameters:
. order - the memory access order with which to release the chunk
*/
template <typename F>
inline void MemoryChunk<F>::release(std::memory_order order) noexcept {
  this->claimed.clear(order);
  return;
}

/*
  MemoryChunk::try_claim - attempt to claim a chunk

  Input Parameters:
. size  - size (in elements) to get

  Returns:
. success - true if you have successfully claimed this block, false otherwise

  Notes:
  The block may be larger than the requested size
*/
template <typename F>
inline bool MemoryChunk<F>::try_claim(size_type size) noexcept {
  return (size <= this->size) && !this->claimed.test_and_set(std::memory_order_seq_cst);
}

/*
  MemoryChunk::force_claim - claim a chunk, error if unsuccessful

  Input Parameters:
. order - the memory access order with which to try and flip the claimed flag

  Notes:
  Errors if claiming was not successful.
*/
template <typename F>
inline PetscErrorCode MemoryChunk<F>::force_claim(std::memory_order order) noexcept {
  PetscFunctionBegin;
  // this should not fail!
  PetscCheck(!this->claimed.test_and_set(order), PETSC_COMM_SELF, PETSC_ERR_PLIB, "Failed to claim memory chunk of size %zu", this->size);
  PetscFunctionReturn(0);
}

// A "memory block" manager, which owns the pointer to a particular memory range. Retrieving
// and restoring a block is thread-safe (so may be used by multiple device streams).
template <typename T, typename AllocatorType>
class MemoryBlock {
public:
  using value_type      = T;
  using allocator_type  = AllocatorType;
  using chunk_type      = MemoryChunk<typename allocator_type::flag_type>;
  using chunk_list_type = std::deque<chunk_type>;
  using size_type       = typename chunk_type::size_type;

  MemoryBlock(allocator_type &alloc, size_type s) : allocator_(alloc), size_(s) {
    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF, alloc.allocate(&mem_, s));
    if (PetscUnlikelyDebug(!mem_)) SETERRABORT(PETSC_COMM_SELF, PETSC_ERR_MEM, "Failed to allocate memory block of size %zu", s);
    PetscFunctionReturnVoid();
  }

  ~MemoryBlock() noexcept(noexcept(std::is_nothrow_destructible<chunk_list_type>::value)) {
    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF, this->destroy(nullptr));
    PetscFunctionReturnVoid();
  }

  /* --- utility accessors --- */
  size_type size() const noexcept { return size_; }
  size_type bytes() const noexcept { return sizeof(value_type) * size(); }
  size_type num_chunks() const noexcept { return chunks_.size(); }

  /* --- actual functions --- */
  template <typename StreamType>
  PETSC_NODISCARD PetscErrorCode try_get_chunk(size_type, T **, StreamType) noexcept;
  template <typename StreamType>
  PETSC_NODISCARD PetscErrorCode try_restore_chunk(T **, StreamType) noexcept;
  template <typename StreamType>
  PETSC_NODISCARD PetscErrorCode destroy(StreamType) noexcept;
  PETSC_NODISCARD bool           owns_pointer(T *) const noexcept;

private:
  value_type     *mem_ = nullptr;
  allocator_type &allocator_;
  const size_type size_;
  chunk_list_type chunks_;
};

/*
  MemoryBlock::try_get_chunk - try to acquire an open chunk from the block

  Input Parameters:
+ size - size (in elements) to get
- ptr  - the pointer you want to get

  Output Parameter:
. ptr - non-null if a chunk was successfully gotten

  Notes:
  If no open chunk is available ptr is unchanged, i.e. NULL.
*/
template <typename T, typename A>
template <typename StreamType>
inline PetscErrorCode MemoryBlock<T, A>::try_get_chunk(size_type size, T **ptr, StreamType stream) noexcept {
  // REVIEW ME: stream could allow us to implement the optimization where we can re-use closed
  // chunks within the same stream. This is tricky in practice however since we would need to
  // "cancel" the block reset in flight. This would probably require the use of an atomic
  // marker flag within the chunk though...
  PetscFunctionBegin;
  if (size <= size_) {
    auto  found  = false;
    auto &result = *ptr;

    // no blocks? make one just for us
    if (chunks_.empty()) PetscCallCXX(chunks_.emplace_back(size));
    for (auto &block : chunks_) {
      if ((found = block.try_claim(size))) {
        // ok found open block of suitable size so claim it. could maybe have shared blocks
        // in the future
        result = mem_ + block.start;
        break;
      }
    }

    if (!found) {
      const auto block_alloced = chunks_.back().start + chunks_.back().size;
      // if we are here can't steal a block so check if the pool has room for a new one
      if ((found = block_alloced + size <= size_)) {
        PetscCallCXX(chunks_.emplace_back(block_alloced, size));
        PetscCall(chunks_.back().force_claim(std::memory_order_relaxed));
        result = mem_ + block_alloced;
      }
    }
    // sets memory to NaN or infinity depending on the type to catch out uninitialized memory
    // accesses.
    if (PetscDefined(USE_DEBUG) && found) PetscCall(allocator_.setCanary(result, size, stream));
  }
  PetscFunctionReturn(0);
}

/*
  MemoryBlock::try_restore_chunk - try to restore a chunk to this MemoryBlock

  Input Parameters:
+ ptr    - ptr to restore
- stream - (optional) stream to restore the pointer on

  Notes:
  ptr is set to nullptr on successful restore, and is unchanged otherwise. If the ptr is owned
  by this MemoryBlock then it is restored on stream, that is, the pointer is not fully restored
  until stream is idle again.
*/
template <typename T, typename A>
template <typename StreamType>
inline PetscErrorCode MemoryBlock<T, A>::try_restore_chunk(T **ptr, StreamType stream) noexcept {
  PetscFunctionBegin;
  if (this->owns_pointer(*ptr)) {
    const auto offset = static_cast<size_type>(*ptr - mem_);
    auto       found  = false;

    for (auto &block : chunks_) {
      if ((found = block.start == offset)) {
        // ok, found ourselves, now set up the destruction mechanism. This may fire at an
        // arbitrary point in the future
        PetscCall(allocator_.call([&] { block.release(); }, stream));
        break;
      }
    }
    PetscAssert(found, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Failed to return %p to block, even though it is within block range [%p, %p)", *ptr, mem_, std::next(mem_, size_));
    *ptr = nullptr;
  }
  PetscFunctionReturn(0);
}

template <typename T, typename A>
template <typename StreamType>
inline PetscErrorCode MemoryBlock<T, A>::destroy(StreamType stream) noexcept {
  PetscFunctionBegin;
  if (mem_) {
    PetscCall(allocator_.deallocate(mem_, stream));
    mem_ = nullptr;
  }
  PetscFunctionReturn(0);
}

/*
  MemoryBock::owns_pointer - returns true if this block owns a pointer, false otherwise
*/
template <typename T, typename A>
inline bool MemoryBlock<T, A>::owns_pointer(T *ptr) const noexcept {
  // each pool is linear in memory, so it suffices to check the bounds
  return (ptr >= mem_) && (ptr < std::next(mem_, size_));
}

namespace detail {

struct DummyAtomicFlag {
  bool _v;

  DummyAtomicFlag() noexcept                                   = default;
  DummyAtomicFlag(const DummyAtomicFlag &)                     = delete;
  DummyAtomicFlag &operator=(const DummyAtomicFlag &)          = delete;
  DummyAtomicFlag &operator=(const DummyAtomicFlag &) volatile = delete;

  PETSC_NODISCARD bool test_and_set(std::memory_order = std::memory_order_seq_cst) noexcept {
    auto old = true;
    std::swap(_v, old);
    return old;
  }

  PETSC_NODISCARD bool test_and_set(std::memory_order = std::memory_order_seq_cst) volatile noexcept {
    const auto tmp = _v;
    _v             = true;
    return tmp;
  }

  void clear(std::memory_order = std::memory_order_seq_cst) noexcept { _v = false; }
  void clear(std::memory_order = std::memory_order_seq_cst) volatile noexcept { _v = false; }
};

template <typename T>
struct real_type {
  using type = T;
};
template <>
struct real_type<PetscScalar> {
  using type = PetscReal;
};

} // namespace detail

template <typename T, typename AtomicFlagType = detail::DummyAtomicFlag>
struct SegmentedMemoryPoolAllocatorBase {
  using value_type      = T;
  using real_value_type = typename detail::real_type<T>::type;
  using flag_type       = AtomicFlagType;

  PETSC_CXX_COMPAT_DECL(PetscErrorCode allocate(value_type **ptr, std::size_t n)) {
    PetscFunctionBegin;
    PetscCall(PetscMalloc1(n, ptr));
    PetscFunctionReturn(0);
  }

  template <typename StreamType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode deallocate(value_type *ptr, StreamType)) {
    PetscFunctionBegin;
    PetscCall(PetscFree(ptr));
    PetscFunctionReturn(0);
  }

  template <typename StreamType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode zero(value_type *ptr, std::size_t n, StreamType)) {
    PetscFunctionBegin;
    PetscCall(PetscArrayzero(ptr, n));
    PetscFunctionReturn(0);
  }

  template <typename StreamType>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode setCanary(value_type *ptr, std::size_t n, StreamType)) {
    using limit_type            = std::numeric_limits<real_value_type>;
    constexpr value_type canary = limit_type::has_signaling_NaN ? limit_type::signaling_NaN() : limit_type::max();

    PetscFunctionBegin;
    for (std::size_t i = 0; i < n; ++i) ptr[i] = canary;
    PetscFunctionReturn(0);
  }

  template <typename U, typename StreamType = std::nullptr_t>
  PETSC_CXX_COMPAT_DECL(PetscErrorCode call(U &&functor, StreamType = StreamType{})) {
    // default implementation immediately runs the functor
    PetscFunctionBegin;
    functor();
    PetscFunctionReturn(0);
  }
};

template <typename MemType, typename AllocType = SegmentedMemoryPoolAllocatorBase<MemType>, std::size_t DefaultChunkSize = 200>
class SegmentedMemoryPool;

// The actual memory pool class. It is in essence just a wrapper for a list of MemoryBlocks.
template <typename MemType, typename AllocType, std::size_t DefaultChunkSize>
class SegmentedMemoryPool : public RegisterFinalizeable<SegmentedMemoryPool<MemType, AllocType, DefaultChunkSize>> {
public:
  using value_type     = MemType;
  using allocator_type = AllocType;
  using block_type     = MemoryBlock<value_type, allocator_type>;
  using pool_type      = std::deque<block_type>;
  using size_type      = typename block_type::size_type;

  explicit SegmentedMemoryPool(allocator_type alloc = allocator_type{}, size_type size = DefaultChunkSize) noexcept(noexcept(std::is_nothrow_default_constructible<pool_type>::value)) : allocator_(std::move(alloc)), chunk_size_(size) { }

  PETSC_NODISCARD PetscErrorCode finalize_() noexcept;
  PETSC_NODISCARD PetscErrorCode register_finalize_() noexcept;
  template <typename StreamType = std::nullptr_t>
  PETSC_NODISCARD PetscErrorCode get(PetscInt, MemType **, StreamType = StreamType{}) noexcept;
  template <typename StreamType = std::nullptr_t>
  PETSC_NODISCARD PetscErrorCode release(MemType **, StreamType = StreamType{}) noexcept;
  template <typename StreamType = std::nullptr_t>
  PETSC_NODISCARD PetscErrorCode pruneEmptyBlocks(StreamType = StreamType{}) noexcept;
  PETSC_NODISCARD PetscErrorCode setChunkSize(size_type) noexcept;

private:
  allocator_type allocator_;
  size_type      chunk_size_;
  pool_type      pool_;

  PETSC_NODISCARD PetscErrorCode make_block_(size_type size) noexcept {
    PetscFunctionBegin;
    PetscCallCXX(pool_.emplace_back(allocator_, std::max(size, chunk_size_)));
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode make_block_() noexcept {
    PetscFunctionBegin;
    PetscCall(make_block_(chunk_size_));
    PetscFunctionReturn(0);
  }
};

/*
  SegmentedMemoryPool::finalize_ - clears the internal memory pool and cleans up

  Notes:
  This routine is automatically registerd for and called from PetscFinalize()
*/
template <typename MemType, typename AllocType, std::size_t DefaultChunkSize>
inline PetscErrorCode SegmentedMemoryPool<MemType, AllocType, DefaultChunkSize>::finalize_() noexcept {
  PetscFunctionBegin;
  PetscCallCXX(pool_.clear());
  PetscCallCXX(pool_.shrink_to_fit());
  chunk_size_ = DefaultChunkSize;
  PetscFunctionReturn(0);
}

/*
  SegmentedMemoryPool::register_finalize_ - initializes the memory pool when it is registered
  for the first time

  Notes:
  Creates the first MemoryBlock and registers the pool for finalization in PetscFinalize()
*/
template <typename MemType, typename AllocType, std::size_t DefaultChunkSize>
inline PetscErrorCode SegmentedMemoryPool<MemType, AllocType, DefaultChunkSize>::register_finalize_() noexcept {
  PetscFunctionBegin;
  PetscCall(make_block_());
  PetscFunctionReturn(0);
}

/*
  SegmentedMemoryPool::get - get an allocation from the memory pool

  Input Parameters:
+ sizein - size (in elements) of the allocation requested
- ptr    - pointer to fill

  Output Parameters:
. ptr - the filled pointer

  Notes:
  This routine will "always" succeed, as in, if the memory pool does not have enough available
  memory to fulfill the allocation request it will allocate new chunks to satisfy the requirement.
*/
template <typename MemType, typename AllocType, std::size_t DefaultChunkSize>
template <typename StreamType>
inline PetscErrorCode SegmentedMemoryPool<MemType, AllocType, DefaultChunkSize>::get(PetscInt sizein, MemType **ptr, StreamType stream) noexcept {
  const auto size = static_cast<size_type>(sizein);

  PetscFunctionBegin;
  PetscAssert(sizein >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Requested memory amount (%" PetscInt_FMT ") must be >= 0", sizein);
  *ptr = nullptr;
  if (!sizein) PetscFunctionReturn(0);
  PetscCall(this->register_finalize());
  for (auto &block : pool_) {
    PetscCall(block.try_get_chunk(size, ptr, stream));
    if (*ptr) PetscFunctionReturn(0);
  }

  PetscCall(PetscInfo(nullptr, "Could not find an open block in the pool (%zu blocks) (requested size %zu), allocating new block\n", pool_.size(), size));
  // if we are here we couldn't find an open block in the pool, so make a new block
  PetscCall(make_block_(size));
  // and assign it
  PetscCall(pool_.back().try_get_chunk(size, ptr, stream));
  PetscAssert(*ptr, PETSC_COMM_SELF, PETSC_ERR_MEM, "Failed to get a suitable memory chunk (of size %zu) from newly allocated memory block (size %zu)", size, pool_.back().size());
  PetscFunctionReturn(0);
}

/*
  SegmentedMemoryPool::release - release a pointer back to the memory pool

  Input Parameters:
+ ptr    - the pointer to return
- stream - (optional) the stream to return the pointer on

  Output Parameters:
. ptr - set to NULL if the pointer was successfully returned

  Notes:
  This routine is coherent on stream (assuming the pool owns ptr). That is, the release is
  guaranteed to have occured if the user synchronizes stream.

  Tries to prune the memory pool each time, perhaps this can be ammended to only do so over a
  set limit...
*/
template <typename MemType, typename AllocType, std::size_t DefaultChunkSize>
template <typename StreamType>
inline PetscErrorCode SegmentedMemoryPool<MemType, AllocType, DefaultChunkSize>::release(MemType **ptr, StreamType stream) noexcept {
  PetscFunctionBegin;
  // nobody owns a nullptr, and if they do then they have bigger problems
  if (!*ptr) PetscFunctionReturn(0);
  for (auto &block : pool_) {
    PetscCall(block.try_restore_chunk(ptr, stream));
    if (!*ptr) break;
  }
  PetscFunctionReturn(0);
}

/*
  SegmentedMemoryPool::pruneEmptyBlocks - try to prune empty memory blocks from the pool

  Input Parameter:
. stream - Stream to use to deallocate memory blocks (may be NULL)

  Notes:
  This may block on stream, so is not called by default anywhere else
*/
template <typename MemType, typename AllocType, std::size_t DefaultChunkSize>
template <typename StreamType>
inline PetscErrorCode SegmentedMemoryPool<MemType, AllocType, DefaultChunkSize>::pruneEmptyBlocks(StreamType stream) noexcept {
  PetscFunctionBegin;
  // try to prune the pool in case of large allocations
  while (pool_.size() > 1) {
    auto &end = pool_.back();

    if (end.num_chunks() == 0) {
      PetscCall(PetscInfo(nullptr, "Freeing empty block of size %zu from pool\n", end.size()));
      PetscCall(end.destroy(stream));
      PetscCallCXX(pool_.pop_back());
    } else {
      break;
    }
  };
  PetscFunctionReturn(0);
}

template <typename MemType, typename AllocType, std::size_t DefaultChunkSize>
inline PetscErrorCode SegmentedMemoryPool<MemType, AllocType, DefaultChunkSize>::setChunkSize(size_type size) noexcept {
  PetscFunctionBegin;
  chunk_size_ = size;
  PetscFunctionReturn(0);
}

} // namespace Impl

} // namespace Device

} // namespace Petsc

#endif // PETSC_SEGMENTEDMEMPOOL_HPP
