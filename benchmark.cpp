// Compile with:
// g++-9 --std=c++17 -O3 -march=native -I benchmarks -I .. -I ../MPMCQueue/include/rigtorp -I ../folly/ -I ~/work/quoine/Hikari/build/gcc-Debug/thirdparty/folly/ -DNDEBUG benchmark.cpp ../folly/folly/detail/Futex.cpp ../folly/folly/synchronization/ParkingLot.cpp -o bench

#include <queue>
#include <algorithm>
#include <functional>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <future>
#include <chrono>
#include <cassert>
#include <iostream>
#include <iomanip>

#include <sys/utsname.h>

// outside of diagnostic push because occur in blockingconcurrentqueue's instantiations
#pragma GCC diagnostic ignored "-Wshadow"
#pragma GCC diagnostic ignored "-Wextra"

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsign-conversion"
#pragma GCC diagnostic ignored "-Wconversion"

#ifdef WITH_BOOST_FIBER
#include <boost/fiber/bounded_channel.hpp>
#endif

#include <boost/lockfree/queue.hpp>
#include "concurrentqueue/blockingconcurrentqueue.h" // moodycamel
#include "readerwriterqueue/readerwriterqueue.h" // moodycamel
#include "MPMCQueue.h" //https://github.com/rigtorp/MPMCQueue.git
#include "SPSCQueue.h" //https://github.com/rigtorp/SPSCQueue.git
#include <folly/MPMCQueue.h>
#include <folly/Function.h>

#pragma GCC diagnostic pop

/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////
/////////////////////////////////////////////////////////////////////////////

// synchronized-queue capacity
// NB: mpmc-bounded-queue requires power of two capacity
static const size_t g_capacity = 4096;

// will pump this many elements through the queue
static const size_t g_num_elements = g_capacity * std::thread::hardware_concurrency();

namespace mt
{

/////////////////////////////////////////////////////////////////////////////
struct timer
{
    /// returns the time elapsed since timer's instantiation, in seconds.
    /// To reset: `my_timer = mt::timer{}`.
    operator double() const
    {
        return (double)std::chrono::duration_cast<std::chrono::nanoseconds>(
            clock_t::now() - start_timepoint).count() * 1e-9;
    }
private:
    using clock_t = std::chrono::steady_clock;
    clock_t::time_point start_timepoint = clock_t::now();
};


/////////////////////////////////////////////////////////////////////////////
/// \brief Can be used as alternative to std::mutex.
/// It is typically faster than std::mutex, yet does not aggressively max-out the CPUs.
/// NB: may cause thread starvation in some scenarios.
class atomic_lock
{
    std::atomic_flag m_flag = ATOMIC_FLAG_INIT;
public:

    atomic_lock() = default;

    // non-copyable and non-movable, as with std::mutex

               atomic_lock(const atomic_lock&) = delete;
    atomic_lock& operator=(const atomic_lock&) = delete;

               atomic_lock(atomic_lock&&) = delete;
    atomic_lock& operator=(atomic_lock&&) = delete;

    /////////////////////////////////////////////////////////////////////////
    bool try_lock() noexcept
    {
        return !m_flag.test_and_set(std::memory_order_acquire);
    }

    void lock() noexcept
    {
        while (m_flag.test_and_set(std::memory_order_acquire))
          std::this_thread::yield();
    }

    void unlock() noexcept
    {
        m_flag.clear(std::memory_order_release);
    }
};


/////////////////////////////////////////////////////////////////////////////
/// A minimalistic blocking, optionally-bounded synchronized-queue.
/// For testing/comparison/benchmarking. not for production.
/// Only requires MoveAssigneable from value_type
template <typename T, class BasicLockable = std::mutex /*or mt::atomic_lock*/>
class naive_synchronized_queue
{
public:
    using value_type = T;

    naive_synchronized_queue(size_t capacity_ = size_t(-1))
        : m_capacity{ capacity_ == 0 ? 1 : capacity_ }
    {}

    /////////////////////////////////////////////////////////////////////////
    template <typename... args_t>
    void push(args_t&&... args)
    {
        // NB: guards are to prevent thread starvation in
        // MPSC and SPMC scenarios when used with atomic_lock
        guard_t guard{ m_push_mutex };
        lock_t lock{ m_mutex };

        m_can_push.wait(
            lock, [this]{ return m_queue.size() < m_capacity; });

        m_queue.emplace(std::forward<args_t>(args)...);

        lock.unlock();
        m_can_pop.notify_one();
    }

    /////////////////////////////////////////////////////////////////////////
    value_type pop()
    {
        guard_t guard{ m_pop_mutex };
        lock_t lock{ m_mutex };

        m_can_pop.wait(
            lock, [this]{ return !m_queue.empty() ;});

        value_type ret = std::move(m_queue.front());
        m_queue.pop();

        lock.unlock();
        m_can_push.notify_one();
        return ret;
    }

    /////////////////////////////////////////////////////////////////////////
private:
    using lock_t = std::unique_lock<BasicLockable>;
    using guard_t = std::lock_guard<BasicLockable>;
    using queue_t = std::queue<value_type>;

    using condvar_t = typename std::conditional<
        std::is_same<BasicLockable, std::mutex>::value,
            std::condition_variable,
            std::condition_variable_any >::type;

         size_t m_capacity;
        queue_t m_queue;
  BasicLockable m_mutex;
  BasicLockable m_push_mutex;
  BasicLockable m_pop_mutex;
      condvar_t m_can_push;
      condvar_t m_can_pop;
}; // naive_synchronized_queue
static void min_sleep()
{
    std::this_thread::sleep_for(
            std::chrono::nanoseconds(1));
    // NB: sleep a minimal amount to avoid hogging the CPU.
    // (the actual time is greater than 1ns; depends on the CPU scheduler)
}

} // namespace mt

/////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////
// Normalizing interface to the various synchronized-queue implementations:
//
//
// template<typename T, typename Queue>
// void push(T value); // blocking
//
// template<typename T, typename Queue>
// T pop(); // blocking
//
// Non-blocking calls are wrapped into a busy-wait loops
// (e.g. boost::lockfree below)

template <typename T, typename lock_t, typename... args_t>
auto push(mt::naive_synchronized_queue<T, lock_t>& q, args_t&&... args) -> decltype((void)q.push(T{}))
{
    q.push(std::forward<args_t>(args)...);
}

template<typename T, typename... args_t>
void push(moodycamel::ConcurrentQueue<T>& q, args_t&&... args)
{
    while (!q.try_enqueue(std::forward<args_t>(args)...))
        std::this_thread::yield();
}

template<typename T, typename... args_t>
void push(moodycamel::BlockingConcurrentQueue<T>& q, args_t&&... args)
{
    while (!q.try_enqueue(std::forward<args_t>(args)...))
        std::this_thread::yield();
}

template<typename T, typename... args_t>
void push(moodycamel::ReaderWriterQueue<T>& q, args_t&&... args)
{
    while (!q.try_enqueue(std::forward<args_t>(args)...))
        std::this_thread::yield();
}

template<typename T, typename... args_t>
void push(moodycamel::BlockingReaderWriterQueue<T>& q, args_t&&... args)
{
    while (!q.try_enqueue(std::forward<args_t>(args)...))
        std::this_thread::yield();
}

template <typename T, typename... args_t>
void push(rigtorp::MPMCQueue<T>& q, args_t&&... args)
{
    q.emplace(std::forward<args_t>(args)...);
}

template <typename T, typename... args_t>
void push(rigtorp::SPSCQueue<T>& q, args_t&&... args)
{
    q.emplace(std::forward<args_t>(args)...);
}

template<typename T, typename... args_t>
void push(folly::MPMCQueue<T>& q, args_t&&... args)
{
    while (!q.write(std::forward<args_t>(args)...))
        std::this_thread::yield();
}

/////////////////////////////////////////////////////////////////////////////

template<typename T, typename Queue>
auto pop(Queue& q) -> decltype(T{ q.pop() })
{
    return q.pop();
}

template<typename T>
T pop(boost::lockfree::queue<T>& q)
{
    T elem{};
    while (!q.pop(elem))
        std::this_thread::yield();
    return elem;
}

template<typename T>
T pop(moodycamel::ConcurrentQueue<T>& q)
{
    auto item = q.try_dequeue();
    for (; !item; item = q.try_dequeue())
        std::this_thread::yield();
    return std::move(*item);
}

template<typename T>
T pop(moodycamel::BlockingConcurrentQueue<T>& q)
{
    return q.wait_dequeue();
}

template<typename T>
T pop(moodycamel::ReaderWriterQueue<T>& q)
{
    T item{};
    while (!q.try_dequeue(item))
        std::this_thread::yield();
    return item;
}

template<typename T>
T pop(moodycamel::BlockingReaderWriterQueue<T>& q)
{
    T item{};
    q.wait_dequeue(item);
    return item;
}

template <typename T>
T pop(rigtorp::SPSCQueue<T>& q)
{
    auto front = q.front();
    for (; !front; front = q.front())
        std::this_thread::yield();
    T item = std::move(*front);
    q.pop();
    return item;
}

template <typename T>
T pop(folly::MPMCQueue<T>& q)
{
  T item{};
  while (!q.read(item))
      std::this_thread::yield();
  return item;
}

/////////////////////////////////////////////////////////////////////////////

/////////////////////////////////////////////////////////////////////////////

/// Pump `num_elems` through the queue using specified number of pushing and popping threads
/// @return throughput (items pumped through the queue per second)
template<typename T, class Queue>
double get_throughput(
          Queue& queue,
          size_t num_pushers,
          size_t num_poppers,
          size_t num_elems)
{
    // verify that num_pushers and num_poppers both divide num_elems
    assert(num_elems/num_pushers*num_pushers == num_elems);
    assert(num_elems/num_poppers*num_poppers == num_elems);

    std::vector<std::future<long>> pushers{ num_pushers };
    std::vector<std::future<long>> poppers{ num_poppers };

    std::string muh_string = "asdaasdasdasdasdsdfgfdgljgkjsdfg;ksdglksdfgjd;lkfgjsd;fkghjhljhasdfjkhksdfhkasdhfkkasdhfkasdkkaskdfjysdufaksdfkasdfbkjs";
    std::vector<std::vector<std::string>> strings(num_pushers);
    for (auto& v : strings) {
      v.reserve(num_elems / num_pushers);
      for (size_t ii = 0; ii != num_elems / num_pushers; ++ii)
        v.push_back(muh_string);
    }

    mt::timer timer;

    // launch pushing tasks
    size_t ii = 0;
    for (auto& p : pushers) {
        p = std::async(std::launch::async, [&, ii]
            {
                long total = 0;
                for (size_t j = 0; j < num_elems/num_pushers; ++j) {
                    push<T>(queue, [asd = std::move(strings[ii][j])] () noexcept { std::cout << asd << std::endl; });
                    ++total;
                }
                return total;
            });
        ++ii;
    }

    // launch popping tasks
    for (auto& p : poppers) {
        p = std::async(std::launch::async, [&]
            {
                long total = 0;
                for(size_t j = 0; j < num_elems/num_poppers; j++) {
                    pop<T>(queue);
                    ++total;
                }
                return total;
            });
    }

    // NB: provided that the time complete the tasks is much greater
    // than the time to spawn the tasks, the bootstrap time can be ignored.

    // wait for the workers thread to finish and aggregate the subtotals

    long total_pushed = 0;
    for(auto& fut : pushers) {
        total_pushed += fut.get();
    }

    long total_popped = 0;
    for(auto& fut : poppers) {
        total_popped += fut.get();
    }

    if(   total_pushed != total_popped
       || total_pushed != (long)num_elems)
    {
        std::cerr << "Problem with queue: pushed: "
                  << total_pushed
                  << "; popped:"
                  << total_popped << "\n";
        assert(false);
    }

    return double(num_elems)/timer;
}

template <typename F>
double get_harmonic_mean(F&& callable, size_t times)
{
    double s = 0;
    for (size_t i = 0; i < times; ++i)
        s += 1.0 / callable();
    return double(times)/s;
}

// Do multiple runs of a single scenario; compute harmonic mean
// of the throughputs and report to cerr.
template <typename T, class Queue>
void test_scenario(
          Queue& queue,
          size_t num_pushers,
          size_t num_poppers,
          size_t num_elems)
{
    auto get_throughput_once = [&queue, num_pushers, num_poppers, num_elems] {
        return get_throughput<T>(queue, num_pushers, num_poppers, num_elems);
    };

    // aggregate over a few runs
    const double throughput = get_harmonic_mean(get_throughput_once, 25);

    const auto s = std::to_string(num_pushers)
           + "/" + std::to_string(num_poppers);

    std::cerr << std::setw(8) << std::left << s
              << std::setw(8) << std::left << std::setprecision(4) << throughput / 1e6
              << std::setw(0) ;

    size_t bar_size = size_t(throughput/1e5)+1;
    size_t bar_capacity = 100;
    for (size_t i = 0; i < std::min(bar_capacity, bar_size); i++) {
        std::cerr << "*";
    }

    std::cerr << (bar_size > bar_capacity ? "..." : "") << std::endl;
}

/////////////////////////////////////////////////////////////////////////////

template <typename T, class Queue>
void test_scenarios(Queue& queue)
{
    size_t num_cores = std::thread::hardware_concurrency();
    size_t half_cores = num_cores == 1 ? 1 : num_cores/2;
    size_t num_elems = g_num_elements;

    test_scenario<T>(queue, 1UL, 1UL,             num_elems / 4);
    test_scenario<T>(queue, 1UL, num_cores - 1UL, num_elems / 16 * (num_cores - 1));
    test_scenario<T>(queue, 2UL, num_cores - 2UL, num_elems / 16 * (num_cores - 2));
    test_scenario<T>(queue, 3UL, num_cores - 3UL, num_elems / 32 * 3 * (num_cores - 3));
    test_scenario<T>(queue, 4UL, 2 * num_cores,   num_elems / 4 * num_cores);
}

template <typename T>
void test_scenarios(rigtorp::SPSCQueue<T>& queue)
{
    size_t num_elems = g_num_elements;

    test_scenario<T>(queue, 1UL, 1UL, num_elems/4);
}

template <typename T>
void test_scenarios(moodycamel::ReaderWriterQueue<T>& queue)
{
    size_t num_elems = g_num_elements;

    test_scenario<T>(queue, 1UL, 1UL, num_elems/4);
}

template <typename T>
void test_scenarios(moodycamel::BlockingReaderWriterQueue<T>& queue)
{
    size_t num_elems = g_num_elements;

    test_scenario<T>(queue, 1UL, 1UL, num_elems/4);
}

/////////////////////////////////////////////////////////////////////////////

// Queue has a constructor accepting capacity.
template<typename T, typename Queue>
void test(const std::string& label, Queue*)
{
    std::cerr << "\n" << label << "\n";
    Queue queue{ g_capacity };
    test_scenarios<T>(queue);
}

/////////////////////////////////////////////////////////////////////////////

int main()
{
    struct utsname uts;
    uname(&uts);

    std::cerr << "hardware concurrency: " << std::thread::hardware_concurrency()
              << "\nplatform:" << uts.sysname << " " << uts.release
              << "\nqueue capacity:" << g_capacity
              << "\nelements to pump: " << g_num_elements
              << "\ncolumns: producers/consumers | throughput(M/s) | bar"
              << std::endl;

    using type = folly::Function<void()>;

#define TEST(...) test<type>(#__VA_ARGS__, reinterpret_cast<__VA_ARGS__*>(NULL));

    TEST( mt::naive_synchronized_queue<type, std::mutex> );
    TEST( mt::naive_synchronized_queue<type, mt::atomic_lock> );

    TEST( moodycamel::ConcurrentQueue<type> );
    TEST( moodycamel::BlockingConcurrentQueue<type> );

    TEST( moodycamel::ReaderWriterQueue<type> );
    TEST( moodycamel::BlockingReaderWriterQueue<type> );

    TEST( rigtorp::MPMCQueue<type> );
    TEST( rigtorp::SPSCQueue<type> );
    TEST( folly::MPMCQueue<type> );

    return 0;
}
