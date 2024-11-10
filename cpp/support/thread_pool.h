/*!
 * Copyright (c) 2023 by Contributors
 * \file support/thread_pool.h
 * \brief Thread pool.
 */
#ifndef MLC_LLM_SUPPORT_THREAD_POOL_H_
#define MLC_LLM_SUPPORT_THREAD_POOL_H_

#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace mlc {
namespace llm {

/*!
 * \brief A thread pool implementation for parallel task execution.
 *
 * ThreadPool manages a pool of worker threads that can execute tasks asynchronously.
 * Tasks are submitted to a queue and executed by available threads from the pool.
 * The pool automatically handles thread synchronization and task distribution.
 */
class ThreadPool {
 public:
  /*!
   * \brief Construct a new thread pool with the specified number of threads.
   * \param num_threads Number of worker threads to create. Defaults to hardware concurrency.
   * \note The pool starts the worker threads immediately upon construction.
   */
  ThreadPool(size_t num_threads = std::thread::hardware_concurrency()) {
    // Initialize thread pool with num_threads threads
    for (size_t i = 0; i < num_threads; ++i) {
      workers_.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            // Lock queue while waiting for new task
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_condition_.wait(lock, [this] { return shutdown_ || !task_queue_.empty(); });

            // Exit thread if shutdown and queue is empty
            if (shutdown_ && task_queue_.empty()) return;

            // Get task from queue
            task = std::move(task_queue_.front());
            task_queue_.pop();
          }
          // Execute task outside the lock to allow other threads to get new tasks
          task();
        }
      });
    }
  }

  /*!
   * \brief Add a new task to be executed by the thread pool.
   * \param task Function object representing the task to execute.
   * \note Tasks are executed in FIFO order but may complete in any order.
   */
  void AddTask(std::function<void()> task) {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      if (shutdown_) {
        throw std::runtime_error("Cannot add task to stopped ThreadPool");
      }
      task_queue_.push(std::move(task));
    }
    queue_condition_.notify_one();  // Wake up one thread to execute task
  }

  /*!
   * \brief Destructor that ensures graceful shutdown of the thread pool.
   *
   * Sets shutdown flag and waits for all threads to complete their current tasks
   * before destroying the pool. Any remaining tasks in the queue will be executed
   * before shutdown completes.
   */
  ~ThreadPool() {
    {
      std::unique_lock<std::mutex> lock(queue_mutex_);
      shutdown_ = true;
    }
    queue_condition_.notify_all();  // Wake up all threads so they can exit
    for (std::thread& worker : workers_) {
      if (worker.joinable()) worker.join();  // Wait for thread to finish
    }
  }

  // Prevent copying or moving of the thread pool
  ThreadPool(const ThreadPool&) = delete;
  ThreadPool(ThreadPool&&) = delete;
  ThreadPool& operator=(const ThreadPool&) = delete;
  ThreadPool& operator=(ThreadPool&&) = delete;

 private:
  /*! \brief Thread container */
  std::vector<std::thread> workers_;
  /*! \brief Task queue */
  std::queue<std::function<void()>> task_queue_;
  /*! \brief Mutex to protect task queue */
  std::mutex queue_mutex_;
  /*! \brief Condition variable for thread synchronization */
  std::condition_variable queue_condition_;
  /*! \brief Flag to indicate thread pool shutdown */
  bool shutdown_ = false;
};

}  // namespace llm
}  // namespace mlc

#endif  // MLC_LLM_SUPPORT_THREAD_POOL_H_
