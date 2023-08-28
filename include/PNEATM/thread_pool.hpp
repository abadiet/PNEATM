#ifndef THREAD_POOL_HPP
#define THREAD_POOL_HPP

#include <thread>
#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <functional>
#include <future>

namespace pneatm {

/**
 * @brief A template class representing a thread pool.
 * @tparam T_Return Template typename of the functions's return type, assuming they all return the same type.
 */
template <typename T_Return>
class ThreadPool {
public:
    /**
     * @brief Constructor for the ThreadPool class.
     * @param numThreads The number of threads. (default is 0 which default to the number of cores)
     */
    ThreadPool (unsigned int numThreads = 0);

    /**
     * @brief Destructor for the ThreadPool class.
     */
    ~ThreadPool ();

    /**
     * @brief Add a task to the queue.
     * @tparam Func The function type.
     * @tparam Args Variadic template for the function's arguments types.
     * @param func The function to be added to the queue.
     * @param args The function's arguments.
     * @return The std::future object of the function's return.
     */
    template <typename Func, typename... Args>
    std::future<T_Return> enqueue (Func&& func, Args&&... args);

private:
    std::vector<std::thread> workers;
    std::queue<std::packaged_task<T_Return ()>> tasks;

    std::mutex queueMutex;
    std::condition_variable condition;
    bool stop = false;
};

template <typename T_Return>
ThreadPool<T_Return>::ThreadPool (unsigned int numThreads) {
    if (numThreads <= 0) {
        // defaulting to the number of core
        numThreads = std::thread::hardware_concurrency ();
    }

    workers.reserve (numThreads);

    for (unsigned int k = 0; k < numThreads; k++) {
        // add a worker
        workers.emplace_back ([&] () {
            // the worker repeatedly checks for tasks in the queue and executes them
            while (true) {
                std::packaged_task<T_Return ()> task;
                {
                    std::unique_lock<std::mutex> lock (queueMutex);
                    condition.wait (lock, [&] { return !tasks.empty () || stop; });
                    if (stop && tasks.empty ()) {
                        return;
                    }
                    task = std::move (tasks.front ());
                    tasks.pop ();
                }
                task ();
            }
        });
    }

    workers.shrink_to_fit ();
}

template <typename T_Return>
ThreadPool<T_Return>::~ThreadPool () {
    {
        std::unique_lock<std::mutex> lock (queueMutex);
        stop = true;
    }
    condition.notify_all ();
    for (std::thread& worker : workers) {
        worker.join ();
    }
}

template <typename T_Return>
template <typename Func, typename... Args>
std::future<T_Return> ThreadPool<T_Return>::enqueue (Func&& func, Args&&... args) {
    // create the package task
    std::packaged_task<T_Return ()> task (
        std::bind (std::forward<Func> (func), std::forward<Args> (args)...)
    );

    // get the future result
    std::future<T_Return> result = task.get_future ();

    // add the task
    {
        std::unique_lock<std::mutex> lock (queueMutex);
        tasks.emplace (std::move (task));
    }

    // there is a new task
    condition.notify_one ();

    return result;
}

}

#endif  // THREAD_POOL_HPP