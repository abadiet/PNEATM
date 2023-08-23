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

template <typename T_Return>
class ThreadPool {
public:
    ThreadPool (unsigned int numThreads = 0);
    ~ThreadPool ();

    template <typename Func, typename... Args>
    std::future<T_Return> enqueue(Func&& func, Args&&... args);

    void waitAllTasks ();

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
    for (unsigned int k = 0; k < numThreads; k++) {
        // add a worker
        workers.emplace_back ([this] () {
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
    condition.notify_one();

    return result;
}

template <typename T_Return>
void ThreadPool<T_Return>::waitAllTasks () {
    std::unique_lock<std::mutex> lock (queueMutex);
    condition.wait (lock, [&] { return tasks.empty (); });
}

}

#endif  // THREAD_POOL_HPP