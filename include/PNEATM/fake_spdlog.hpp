#ifndef FAKE_SPDLOG_HPP
#define FAKE_SPDLOG_HPP

#include <PNEATM/utils.hpp>
#include <memory>
#include <string>


namespace spdlog {
    class logger {
    public:
        template<typename... Args>
        void trace(const char* fmt, const Args&... args) {
            UNUSED(fmt);
            UNUSED_PACK(args);
        }

        template<typename... Args>
        void debug(const char* fmt, const Args&... args) {
            UNUSED(fmt);
            UNUSED_PACK(args);
        }

        template<typename... Args>
        void info(const char* fmt, const Args&... args) {
            UNUSED(fmt);
            UNUSED_PACK(args);
        }

        template<typename... Args>
        void warn(const char* fmt, const Args&... args) {
            UNUSED(fmt);
            UNUSED_PACK(args);
        }

        template<typename... Args>
        void error(const char* fmt, const Args&... args) {
            UNUSED(fmt);
            UNUSED_PACK(args);
        }

        template<typename... Args>
        void critical(const char* fmt, const Args&... args) {
            UNUSED(fmt);
            UNUSED_PACK(args);
        }
    };
    enum level { trace, debug, info, warn, error, critical, off };

    class pattern_formatter {};

    class sink {};

    inline std::shared_ptr<logger> stdout_color_mt(const std::string&) {
        return std::make_shared<logger>();
    }

    inline std::shared_ptr<logger> rotating_logger_mt(const std::string&, const std::string&, size_t, size_t) {
        return std::make_shared<logger>();
    }

    inline std::shared_ptr<logger> basic_logger_mt(const std::string&, const std::string&) {
        return std::make_shared<logger>();
    }

    inline void set_pattern(const std::string&) {}

    inline void set_level(level) {}
}


#endif // FAKE_SPDLOG_HPP