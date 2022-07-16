#ifndef PETSC_SEGMENTEDMEMPOOL_HPP
#define PETSC_SEGMENTEDMEMPOOL_HPP

#include <petsc/private/deviceimpl.h>
#include <petsc/private/cpp/macros.hpp>
#include <petsc/private/cpp/type_traits.hpp>
#include <petsc/private/cpp/utility.hpp>
#include <petsc/private/cpp/register_finalize.hpp>

#include <limits>
#include <deque>
#include <vector>

// REVIEW ME:
// REMOVE ME:
#include <petsc/private/cupminterface.hpp>

namespace Petsc {

namespace device {

template <typename T>
class StreamBase {
public:
  using id_type      = int;
  using derived_type = T;

  // needed so that dependent auto works, see veccupmimpl.h for a detailed discussion
  template <typename U = T>
  PETSC_NODISCARD auto get_stream() const noexcept PETSC_DECLTYPE_AUTO_RETURNS(static_cast<const U *>(this)->get_stream_())

    PETSC_NODISCARD id_type get_id() const noexcept {
    return static_cast<const T *>(this)->get_id_();
  }

  template <typename E>
  PETSC_NODISCARD PetscErrorCode record_event(E &&event) const noexcept {
    return static_cast<const T *>(this)->record_event_(std::forward<E>(event));
  }

  template <typename E>
  PETSC_NODISCARD PetscErrorCode wait_for(E &&event) const noexcept {
    return static_cast<const T *>(this)->wait_for_(std::forward<E>(event));
  }

private:
  constexpr StreamBase() noexcept = default;
  friend T;

  struct default_event_type { };
  struct default_stream_type { };

  PETSC_NODISCARD static constexpr default_stream_type get_stream_() noexcept { return default_stream_type{}; }

  PETSC_NODISCARD static constexpr id_type get_id_() noexcept { return 0; }

  template <typename U = T>
  PETSC_NODISCARD static constexpr PetscErrorCode record_event_(typename U::event_type) noexcept {
    return 0;
  }

  template <typename U = T>
  PETSC_NODISCARD static constexpr PetscErrorCode wait_for_(typename U::event_type) noexcept {
    return 0;
  }
};

struct DefaultStream : StreamBase<DefaultStream> {
  using stream_type = typename StreamBase<DefaultStream>::default_stream_type;
  using id_type     = typename StreamBase<DefaultStream>::id_type;
  using event_type  = typename StreamBase<DefaultStream>::default_event_type;
};

} // namespace device

namespace memory {

namespace impl {

template <typename E>
class MemoryChunk {
public:
  using event_type = E;
  using size_type  = std::size_t;

  MemoryChunk(size_type start, size_type size) noexcept : open_(true), size_(size), stream_id_(-1), event_(), start_(start) { }

  explicit MemoryChunk(size_type size) noexcept : MemoryChunk(0, size) { }

  PETSC_NODISCARD size_type constexpr start() const noexcept { return start_; }
  PETSC_NODISCARD size_type constexpr size() const noexcept { return size_; }
  PETSC_NODISCARD size_type constexpr total_offset() const noexcept { return start() + size(); }

  template <typename U>
  PetscErrorCode release(const device::StreamBase<U> *) noexcept;
  template <typename U>
  PETSC_NODISCARD PetscErrorCode claim(const device::StreamBase<U> *, size_type, bool *, bool = false) noexcept;
  template <typename U>
  PETSC_NODISCARD bool can_claim(const device::StreamBase<U> *, size_type, bool) const noexcept;

private:
  bool            open_;
  size_type       size_;
  int             stream_id_;
  event_type      event_;
  const size_type start_;

  template <typename U>
  PETSC_NODISCARD bool stream_compat_(const device::StreamBase<U> *strm) const noexcept {
    return (stream_id_ == -1) || (stream_id_ == strm->get_id());
  }
};

template <typename E>
template <typename U>
inline PetscErrorCode MemoryChunk<E>::release(const device::StreamBase<U> *stream) noexcept {
  NVTX_RANGE;
  PetscFunctionBegin;
  open_      = true;
  stream_id_ = stream->get_id();
  PetscCall(stream->record_event(event_));
  PetscFunctionReturn(0);
}

/*
  MemoryChunk::claim - attempt to claim a particular chunk

  Input Parameters:
+ stream    - the stream on which to attempt to claim
. req_size  - the requested size (in elements) to attempt to claim
- serialize - (optional, false) whether the claimant allows serialization

  Output Parameter:
. success - true if the chunk was claimed, false otherwise
*/
template <typename E>
template <typename U>
inline PetscErrorCode MemoryChunk<E>::claim(const device::StreamBase<U> *stream, size_type req_size, bool *success, bool serialize) noexcept {
  NVTX_RANGE;
  PetscFunctionBegin;
  if ((*success = can_claim(stream, req_size, serialize))) {
    if (serialize && !stream_compat_(stream)) PetscCall(stream->wait_for(event_));
    open_ = false;
    size_ = req_size;
  }
  PetscFunctionReturn(0);
}

/*
  MemoryChunk::can_claim - test whether a particular chunk can be claimed

  Input Parameters:
+ stream    - the stream on which to attempt to claim
. req_size  - the requested size (in elements) to attempt to claim
- serialize - whether the claimant allows serialization

  Output:
. [return] - true if the chunk is claimable given the configuration, false otherwise
*/
template <typename E>
template <typename U>
inline bool MemoryChunk<E>::can_claim(const device::StreamBase<U> *stream, size_type req_size, bool serialize) const noexcept {
  if (open_ && (req_size <= size())) {
    // fully compatible
    if (stream_compat_(stream)) return true;
    // stream wasn't compatible, but could claim if we serialized
    if (serialize) return true;
    // incompatible stream and did not want to serialize
  }
  return false;
}

// A "memory block" manager, which owns the pointer to a particular memory range. Retrieving
// and restoring a block is thread-safe (so may be used by multiple device streams).
template <typename T, typename AllocatorType, typename StreamType>
class MemoryBlock {
public:
  using value_type      = T;
  using allocator_type  = AllocatorType;
  using stream_type     = StreamType;
  using chunk_type      = MemoryChunk<typename stream_type::event_type>;
  using size_type       = typename chunk_type::size_type;
  using chunk_list_type = std::vector<chunk_type>;

  MemoryBlock(allocator_type &alloc, size_type s) noexcept : size_(s), allocator_(alloc) {
    NVTX_RANGE;
    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF, alloc.allocate(&mem_, s));
    PetscAssertAbort(mem_, PETSC_COMM_SELF, PETSC_ERR_MEM, "Failed to allocate memory block of size %zu", s);
    PetscCallAbort(PETSC_COMM_SELF, alloc.zero(mem_, s, nullptr));
    PetscFunctionReturnVoid();
  }

  ~MemoryBlock() noexcept(std::is_nothrow_destructible<chunk_list_type>::value) {
    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF, self_destruct_());
    PetscFunctionReturnVoid();
  }

  MemoryBlock(MemoryBlock &&other) noexcept : mem_(other.mem_), size_(other.size()), chunks_(std::move(other.chunks_)), allocator_(other.allocator_) { other.mem_ = nullptr; }

  MemoryBlock &operator=(MemoryBlock &&other) noexcept {
    PetscFunctionBegin;
    PetscCallAbort(PETSC_COMM_SELF, self_destruct_());
    mem_       = other.mem_;
    size_      = other.size();
    chunks_    = std::move(other.chunks_);
    allocator_ = other.allocator_;
    PetscFunctionReturn(*this);
  }

  // memory blocks are not copyable
  MemoryBlock(const MemoryBlock &)            = delete;
  MemoryBlock &operator=(const MemoryBlock &) = delete;

  /* --- actual functions --- */
  PETSC_NODISCARD PetscErrorCode try_get_chunk(size_type, T **, const stream_type *, bool *) noexcept;
  PETSC_NODISCARD PetscErrorCode try_restore_chunk(T **, const stream_type *, bool *) noexcept;
  PETSC_NODISCARD bool           owns_pointer(T *) const noexcept;

  PETSC_NODISCARD constexpr size_type size() const noexcept { return size_; }
  PETSC_NODISCARD constexpr size_type bytes() const noexcept { return sizeof(value_type) * size(); }
  PETSC_NODISCARD size_type           num_chunks() const noexcept { return chunks_.size(); }

private:
  value_type     *mem_ = nullptr;
  const size_type size_;
  chunk_list_type chunks_;
  allocator_type &allocator_;

  PETSC_NODISCARD PetscErrorCode self_destruct_() noexcept {
    PetscFunctionBegin;
    if (PetscLikely(mem_)) {
      PetscCall(allocator_.deallocate(mem_, nullptr));
      mem_ = nullptr;
    }
    PetscFunctionReturn(0);
  }
};

/*
  MemoryBock::owns_pointer - returns true if this block owns a pointer, false otherwise
*/
template <typename T, typename A, typename S>
inline bool MemoryBlock<T, A, S>::owns_pointer(T *ptr) const noexcept {
  // each pool is linear in memory, so it suffices to check the bounds
  return (ptr >= mem_) && (ptr < std::next(mem_, size()));
}

/*
  MemoryBlock::try_get_chunk - try to get a chunk from this MemoryBlock

  Input Parameters:
+ req_size - the requested size of the allocation (in elements)
. ptr      - ptr to fill
- stream   - stream to fill the pointer on

  Output Parameter:
. success  - true if chunk was gotten, false otherwise

  Notes:
  If the current memory could not satisfy the memory request, ptr is unchanged
*/
template <typename T, typename A, typename S>
inline PetscErrorCode MemoryBlock<T, A, S>::try_get_chunk(size_type req_size, T **ptr, const stream_type *stream, bool *success) noexcept {
  NVTX_RANGE;
  PetscFunctionBegin;
  *success = false;
  if (req_size <= size()) {
    const auto try_create_chunk = [&]() {
      const auto was_empty     = chunks_.empty();
      const auto block_alloced = was_empty ? 0 : chunks_.back().total_offset();

      PetscFunctionBegin;
      if (block_alloced + req_size <= size()) {
        PetscCallCXX(chunks_.emplace_back(block_alloced, req_size));
        PetscCall(chunks_.back().claim(stream, req_size, success));
        *ptr = mem_ + block_alloced;
        if (was_empty) PetscAssert(*success, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Failed to claim chunk (of size %zu) even though block (of size %zu) was empty!", req_size, size());
      }
      PetscFunctionReturn(0);
    };
    const auto try_find_open_chunk = [&](bool serialize = false) {
      PetscFunctionBegin;
      for (auto &chunk : chunks_) {
        PetscCall(chunk.claim(stream, req_size, success, serialize));
        if (*success) {
          *ptr = mem_ + chunk.start();
          break;
        }
      }
      PetscFunctionReturn(0);
    };

    // search previously distributed chunks, but only claim one if it is on the same stream
    // as us
    PetscCall(try_find_open_chunk());

    // if we are here we couldn't reuse one of our own chunks so check first if the pool
    // has room for a new one
    if (!*success) PetscCall(try_create_chunk());

    // try pruning dead chunks off the back, note we do this regardless of whether we are
    // successful
    while (chunks_.back().can_claim(stream, 0, false)) {
      PetscCallCXX(chunks_.pop_back());
      if (chunks_.empty()) {
        // if chunks are empty it implies we have managed to claim (and subsequently destroy)
        // our own chunk twice! something has gone wrong
        PetscAssert(!*success, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Successfully claimed a chunk (of size %zu, from block of size %zu) but have now managed to claim it for a second time (and destroyed it)!", req_size, size());
        break;
      }
    }

    // if previously unsuccessful see if enough space has opened up due to pruning. note that
    // if the chunk list was emptied from the pruning this call must succeed in allocating a
    // chunk, otherwise something is wrong
    if (!*success) PetscCall(try_create_chunk());

    // last resort, iterate over all chunks and see if we can steal one by waiting on the
    // current owner to finish using it
    if (!*success) PetscCall(try_find_open_chunk(true));

    // sets memory to NaN or infinity depending on the type to catch out uninitialized memory
    // accesses.
    if (PetscDefined(USE_DEBUG) && *success) { PetscCall(allocator_.setCanary(*ptr, req_size, stream->get_stream())); }
  }
  PetscFunctionReturn(0);
}

/*
  MemoryBlock::try_restore_chunk - try to restore a chunk to this MemoryBlock

  Input Parameters:
+ ptr     - ptr to restore
- stream  - stream to restore the pointer on

  Output Parameter:
. success - true if chunk was restored, false otherwise

  Notes:
  ptr is set to nullptr on successful restore, and is unchanged otherwise. If the ptr is owned
  by this MemoryBlock then it is restored on stream. The same stream may recieve ptr again
  without synchronization, but other streams may not do so until either serializing or the
  stream is idle again.
*/
template <typename T, typename A, typename S>
inline PetscErrorCode MemoryBlock<T, A, S>::try_restore_chunk(T **ptr, const stream_type *stream, bool *success) noexcept {
  NVTX_RANGE;
  PetscFunctionBegin;
  if ((*success = this->owns_pointer(*ptr))) {
    const auto offset      = static_cast<size_type>((*ptr) - mem_);
    auto       found_block = false;

    for (auto &chunk : chunks_) {
      if ((found_block = chunk.start() == offset)) {
        PetscCall(chunk.release(stream));
        break;
      }
    }
    PetscAssert(found_block, PETSC_COMM_SELF, PETSC_ERR_PLIB, "Failed to return %zu to block, even though it is within block range [%zu, %zu)", reinterpret_cast<uintptr_t>(*ptr), reinterpret_cast<uintptr_t>(mem_), reinterpret_cast<uintptr_t>(std::next(mem_, size())));
    *ptr = nullptr;
  }
  PetscFunctionReturn(0);
}

namespace detail {

template <typename T>
struct real_type {
  using type = T;
};
template <>
struct real_type<PetscScalar> {
  using type = PetscReal;
};

} // namespace detail

template <typename T>
struct SegmentedMemoryPoolAllocatorBase {
  using value_type      = T;
  using real_value_type = typename detail::real_type<T>::type;

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
};

} // namespace impl

template <typename MemType, typename StreamType = device::DefaultStream, typename AllocType = impl::SegmentedMemoryPoolAllocatorBase<MemType>, std::size_t DefaultChunkSize = 256>
class SegmentedMemoryPool;

// The actual memory pool class. It is in essence just a wrapper for a list of MemoryBlocks.
template <typename MemType, typename StreamType, typename AllocType, std::size_t DefaultChunkSize>
class SegmentedMemoryPool : RegisterFinalizeable<SegmentedMemoryPool<MemType, StreamType, AllocType, DefaultChunkSize>> {
  friend class RegisterFinalizeable<SegmentedMemoryPool<MemType, StreamType, AllocType, DefaultChunkSize>>;

public:
  using value_type     = MemType;
  using stream_type    = StreamType;
  using allocator_type = AllocType;
  using block_type     = impl::MemoryBlock<value_type, allocator_type, stream_type>;
  using pool_type      = std::deque<block_type>;
  using size_type      = typename block_type::size_type;

  explicit SegmentedMemoryPool(AllocType alloc = AllocType{}, std::size_t size = DefaultChunkSize) noexcept(std::is_nothrow_default_constructible<pool_type>::value) : allocator_(std::move(alloc)), chunk_size_(size) { }

  PETSC_NODISCARD PetscErrorCode get(PetscInt, MemType **, const stream_type *) noexcept;
  PETSC_NODISCARD PetscErrorCode release(MemType **, const stream_type *) noexcept;

private:
  pool_type      pool_;
  allocator_type allocator_;
  size_type      chunk_size_;

  PETSC_NODISCARD PetscErrorCode register_finalize_() noexcept {
    PetscFunctionBegin;
    PetscCall(make_block_());
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode finalize_() noexcept {
    PetscFunctionBegin;
    PetscCallCXX(pool_.clear());
    chunk_size_ = DefaultChunkSize;
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode make_block_(size_type size) noexcept {
    const auto block_size = std::max(size, chunk_size_);

    PetscFunctionBegin;
    PetscCallCXX(pool_.emplace_back(allocator_, block_size));
    PetscCall(PetscInfo(nullptr, "Allocated new block of size %zu, total %zu blocks\n", block_size, pool_.size()));
    PetscFunctionReturn(0);
  }

  PETSC_NODISCARD PetscErrorCode make_block_() noexcept {
    PetscFunctionBegin;
    PetscCall(make_block_(chunk_size_));
    PetscFunctionReturn(0);
  }
};

/*
  SegmentedMemoryPool::get - get an allocation from the memory pool

  Input Parameters:
+ req_size - size (in elements) to get
. ptr      - the pointer to hold the allocation
- stream   - the stream on which to get the allocation

  Output Parameter:
. ptr - the pointer holding the allocation

  Notes:
  req_size cannot be negative. If req_size if zero, ptr is set to nullptr
*/
template <typename MemType, typename StreamType, typename AllocType, std::size_t DefaultChunkSize>
inline PetscErrorCode SegmentedMemoryPool<MemType, StreamType, AllocType, DefaultChunkSize>::get(PetscInt req_size, MemType **ptr, const StreamType *stream) noexcept {
  NVTX_RANGE;
  const auto size  = static_cast<size_type>(req_size);
  auto       found = false;

  PetscFunctionBegin;
  PetscAssert(req_size >= 0, PETSC_COMM_SELF, PETSC_ERR_ARG_OUTOFRANGE, "Requested memory amount (%" PetscInt_FMT ") must be >= 0", req_size);
  PetscValidPointer(ptr, 2);
  PetscValidPointer(stream, 3);
  *ptr = nullptr;
  if (!req_size) PetscFunctionReturn(0);
  PetscCall(this->register_finalize());
  for (auto &block : pool_) {
    PetscCall(block.try_get_chunk(size, ptr, stream, &found));
    if (PetscLikely(found)) PetscFunctionReturn(0);
  }

  PetscCall(PetscInfo(nullptr, "Could not find an open block in the pool (%zu blocks) (requested size %zu), allocating new block\n", pool_.size(), size));
  // if we are here we couldn't find an open block in the pool, so make a new block
  PetscCall(make_block_(size));
  // and assign it
  PetscCall(pool_.back().try_get_chunk(size, ptr, stream, &found));
  PetscAssert(found, PETSC_COMM_SELF, PETSC_ERR_MEM, "Failed to get a suitable memory chunk (of size %zu) from newly allocated memory block (size %zu)", size, pool_.back().size());
  PetscFunctionReturn(0);
}

/*
  SegmentedMemoryPool::release - release a pointer back to the memory pool

  Input Parameters:
+ ptr    - the pointer to release
- stream - the stream to release it on

  Notes:
  If ptr is not owned by the pool it is unchanged.
*/
template <typename MemType, typename StreamType, typename AllocType, std::size_t DefaultChunkSize>
inline PetscErrorCode SegmentedMemoryPool<MemType, StreamType, AllocType, DefaultChunkSize>::release(MemType **ptr, const StreamType *stream) noexcept {
  NVTX_RANGE;

  PetscFunctionBegin;
  PetscValidPointer(ptr, 1);
  PetscValidPointer(stream, 2);
  // nobody owns a nullptr, and if they do then they have bigger problems
  if (!*ptr) PetscFunctionReturn(0);
  for (auto &block : pool_) {
    auto found = false;

    PetscCall(block.try_restore_chunk(ptr, stream, &found));
    if (PetscLikely(found)) break;
  }
  PetscFunctionReturn(0);
}

} // namespace memory

} // namespace Petsc

#endif // PETSC_SEGMENTEDMEMPOOL_HPP
