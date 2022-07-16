#ifndef PETSC_FLAT_MAP_HPP
#define PETSC_FLAT_MAP_HPP

#include <vector>
#include <algorithm>
#include <functional>

namespace Petsc {

namespace util {

template <class Key, class, class = std::equal_to<Key>, template <class...> class = std::vector>
class flat_map;

template <class Key, class Value, class Equal, template <class...> class Container>
class flat_map {
public:
  using key_type       = Key;
  using mapped_type    = Value;
  using equal_type     = Equal;
  using value_type     = std::pair<const key_type, mapped_type>;
  using container_type = Container<value_type>;
  using size_type      = typename container_type::size_type;
  using iterator       = typename container_type::iterator;
  using const_iterator = typename container_type::const_iterator;

  PETSC_NODISCARD iterator begin() noexcept {
    using std::begin;
    return begin(data_);
  }

  PETSC_NODISCARD const_iterator begin() const noexcept {
    using std::begin;
    return begin(data_);
  }

  PETSC_NODISCARD const_iterator cbegin() const noexcept {
    using std::cbegin;
    return cbegin(data_);
  }

  PETSC_NODISCARD iterator end() noexcept {
    using std::end;
    return end(data_);
  }

  PETSC_NODISCARD const_iterator end() const noexcept {
    using std::end;
    return end(data_);
  }

  PETSC_NODISCARD const_iterator cend() const noexcept {
    using std::cend;
    return cend(data_);
  }

  PETSC_NODISCARD size_type size() const noexcept { return data_.size(); }
  PETSC_NODISCARD size_type capacity() const noexcept { return data_.capacity(); }
  PETSC_NODISCARD bool      empty() const noexcept { return data_.empty(); }

  void clear() noexcept { data_.clear(); }

  // these only have to be valid if called:
  void reserve(size_type n) { data_.reserve(n); }

  mapped_type &operator[](const key_type &k) {
    auto it = find(k);

    if (it != end()) return it->second;
    data_.emplace_back(k, mapped_type{});
    return data_.back().second;
  }

  iterator find(const key_type &k) noexcept {
    return std::find_if(begin(), end(), [&](const value_type &kv) { return equal_to_(kv.first, k); });
  }

  const_iterator find(const key_type &k) const { return const_cast<flat_map *>(this)->find(k); }

  bool erase(const key_type &k) {
    const auto end_it = end();
    auto       it     = std::remove(begin(), end_it, [&](const value_type &kv) { return equal_to_(kv.first, k); });

    if (it == end_it) return false;
    data_.erase(it, end_it);
    return true;
  }

  // classic erase, for iterating:
  iterator erase(const_iterator it) { return data_.erase(it); }

private:
  equal_type     equal_to_;
  container_type data_;
};

} // namespace util

} // namespace Petsc

#endif // PETSC_FLAT_MAP_HPP
