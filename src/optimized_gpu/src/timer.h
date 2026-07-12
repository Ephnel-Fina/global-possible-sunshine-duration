#include <iostream>
#include <iomanip>
#include <chrono>
#include <string>
#include <stack>
#include <vector>
#include <algorithm>
#include <utility>

class Timer
{
private:
    using T = std::chrono::high_resolution_clock;
    using TP = std::chrono::time_point<T>;

    int __id = 0;

    // <tag, start_time>
    std::stack<std::pair<int, std::pair<std::string, TP>>> __time_stack;
    // <<tag, level>, <start_time, end_time>>
    std::vector<std::pair<std::pair<std::string, int>, std::pair<TP, TP>>> __records;

    enum TimeUnit
    {
        S,
        MS
    };

    TimeUnit __time_unit = MS;

    inline void print(std::string tag, TP start, TP end, int level = 0)
    {
        auto duration = std::chrono::duration<double>(end - start);
        for (; level; level--)
            std::cout << "    ";
        std::cout << std::setiosflags(std::ios::fixed) << std::setprecision(3);
        if (__time_unit == S)
            std::cout << tag << ": " << duration.count() << "s." << std::endl;
        else if (__time_unit == MS)
            std::cout << tag << ": " << duration.count() * 1000 << "ms." << std::endl;
        else
            std::cerr << "[Timer] Error: Unknown time unit." << std::endl;
    }

public:
    void tick(std::string tag)
    {
        auto start = T::now();
        __time_stack.push(std::make_pair(__id++, std::make_pair(tag, start)));
        __records.emplace_back();
    }

    void tock()
    {
        auto end = T::now();
        int id = __time_stack.top().first;
        auto start = __time_stack.top().second.second;
        std::string tag = __time_stack.top().second.first;
        std::cout << "[Timer] ";
        print(tag, start, end);
        __records[id] = std::make_pair(std::make_pair(tag, __time_stack.size() - 1), std::make_pair(start, end));
        __time_stack.pop();
    }

    double get_last_time(std::string tag)
    {
        return std::chrono::duration<double>(__records.back().second.second - __records.back().second.first).count();
    }

    void print_records()
    {
        std::cout << "================================================================" << std::endl;
        std::cout << "[Timer] ";
        std::cout << "Time Report" << std::endl;

        auto start_time = T::to_time_t(__records.front().second.first);
        auto end_time = T::to_time_t(__records.back().second.first);

        std::string time_str = std::ctime(&start_time);
        std::cout << "Start time: " << time_str;

        for(auto& record : __records)
        {
            std::string tag = record.first.first;
            int level = record.first.second;
            auto start = record.second.first;
            auto end = record.second.second;

            if (level == 0)
            {
                if (std::chrono::duration<double>(end - start) > std::chrono::seconds(10))
                    __time_unit = S;
                else
                    __time_unit = MS;
            }

            print(tag, start, end, level);
        }

        time_str = std::ctime(&end_time);
        std::cout << "End time: " << time_str << std::endl;
    }

    void merge_records(const Timer& other)
    {
        for(auto& record : other.__records)
        {
            __records.push_back(record);
            __records.back().first.second += __time_stack.size();
            ++__id;
        }
    }
};

inline Timer timer;
