/* *********************************************************************************************************
 * @Author: pjCui
 * @Date: 2024-04-15 13:49:08
 * @LastEditTime: 2024-04-15 14:37:29
 * @FilePath: /2024-4-2/src/Basic/Console/console_RDMA.hpp
 * @Description: 我将Basic/Console/console_V3.hpp 升级为 console_V3_3.hpp
 *               我利用constexpr来减少代码量
 *               新增Test_passed 和 Test_failed 宏定义
 * *********************************************************************************************************/
#pragma once

#include <libgen.h>
#include <omp.h>

#include <cstring>
#include <iostream>

#include "Basic/Log/log.hpp"
#include "Basic/Other/placeholder_CPJ.hpp"
#include "Comm/rdma_env.hpp"
// #include "console_var.hpp"
#define ISCONSOLE 1      // 是否输出到Console
#define ISLOG 0          // 是否输出到Log
#define CONSOLE_COLOR 1  // 带颜色的Console打印
#define GPU_THREAD_ID (threadIdx.x + blockIdx.x * blockDim.x)

#if (CONSOLE_COLOR == 1)
#define ESC_START "\033["
#define ESC_END "\033[0m"
#define COLOR_FATAL "31;40;5m"  // 5:并启用闪烁的文本
#define COLOR_ALERT "31;40;1m"
#define COLOR_CRIT "31;40;1m"
#define COLOR_ERROR \
  "38;2;255;45;0m"  // 31:红色        38表示前景色 2 表示RGB模式
                    // 后面三个分别是RGB的值
#define COLOR_WARN "38;2;255;255;0m"    // 33:黄色 "33;40;1m"
#define COLOR_NOTICE "38;2;255;9;255m"  // 34:蓝色  MAJOR- "34;40;1m"
#define COLOR_CHECK "38;2;18;255;255m"  // 32:青色 "36;48;0m"
#define COLOR_WRITE "38;2;255;127;0m"   // 36:亮棕色 38;2;217;168;27m
#define COLOR_FREES "38;2;35;209;139m"  // 32:绿色
#define COLOR_FINSH "38;2;25;255;4m"    // 32:亮绿色
#define COLOR_INFOS "37;48;1m"  // 37:白色  "37;40;1m"  48对应RGB{30,30,30}
#define COLOR_RATES "33;48;1m"  // 33:黄色
#define COLOR_LOG "38;2;75;0;130;1m"          // 35:紫色
#define COLOR_LOG_GPU "38;5;6m"               // 35:浅蓝色
#define COLOR_TEST_PASSED "38;2;11;193;193m"  //
#define COLOR_TEST_FAILED \
  "38;2;255;127;0m"  // "38;2;255;76;16m"  //"38;2;217;168;27m"
#else
#define ESC_START ""
#define ESC_END ""
#define COLOR_FATAL ""
#define COLOR_ALERT ""
#define COLOR_CRIT ""
#define COLOR_ERROR ""
#define COLOR_WARN ""
#define COLOR_NOTICE ""
#define COLOR_CHECK ""
#define COLOR_WRITE ""
#define COLOR_FREES ""
#define COLOR_FINSH ""
#define COLOR_INFOS ""
#define COLOR_RATES ""
#define COLOR_LOG ""
#define COLOR_LOG_GPU ""
#define COLOR_TEST_PASSED ""
#define COLOR_TEST_FAILED ""
#endif

#define assert_msgsss(condition, format, ...)                         \
  if (true)                                                           \
  {                                                                   \
    if (__builtin_expect(!(condition), 0))                            \
    {                                                                 \
      fprintf(stderr,                                                 \
              ESC_START COLOR_ERROR "<ERROR-%d>:" format              \
                                    " -> T[%u] [%s:%u 行]\n" ESC_END, \
              r_server_id, ##__VA_ARGS__, omp_get_thread_num(),       \
              basename((char *)(__FILE__)), __LINE__);                \
      fflush(stderr);                                                 \
      if constexpr (ISLOG == 1)                                       \
      {                                                               \
        global_logFile().myFlush();                                   \
        global_logFile().myClose();                                   \
      }                                                               \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  }

#define assert_msg(condition, format, ...)                            \
  if (true)                                                           \
  {                                                                   \
    if (__builtin_expect(!(condition), 0))                            \
    {                                                                 \
      fprintf(stderr,                                                 \
              ESC_START COLOR_ERROR "<ERROR-%d>:" format              \
                                    " -> T[%u] [%s:%u 行]\n" ESC_END, \
              r_server_id, ##__VA_ARGS__, omp_get_thread_num(),       \
              basename((char *)(__FILE__)), __LINE__);                \
      fflush(stderr);                                                 \
      if constexpr (ISLOG == 1)                                       \
      {                                                               \
        global_logFile().myFlush();                                   \
        global_logFile().myClose();                                   \
      }                                                               \
      exit(EXIT_FAILURE);                                             \
    }                                                                 \
  }

#define assert_msg_clear(condition, format, ...)                      \
  if (true)                                                           \
  {                                                                   \
    if (__builtin_expect(!(condition), 0))                            \
    {                                                                 \
      fprintf(stderr,                                                 \
              ESC_START COLOR_ERROR "<ERGO-%d>: " format              \
                                    " -> T[%u] [%s:%u 行]\n" ESC_END, \
              r_server_id, ##__VA_ARGS__, omp_get_thread_num(),       \
              basename((char *)__FILE__), __LINE__);                  \
      fflush(stderr);                                                 \
      if constexpr (ISLOG == 1)                                       \
      {                                                               \
        global_logFile().myFlush();                                   \
        global_logFile().myClose();                                   \
      }                                                               \
      goto clear;                                                     \
    }                                                                 \
  }

#define STOP assert_msg(false, "Clear-Stop")

#define Msg_info(format, ...)                                            \
  {                                                                      \
    if constexpr (ISCONSOLE)                                             \
      fprintf(stderr,                                                    \
              ESC_START COLOR_INFOS "[INFOS-%d T-%d]: " format           \
                                    " -> [%s:%u 行]" ESC_END "\n",       \
              r_server_id, omp_get_thread_num(), ##__VA_ARGS__,          \
              basename((char *)__FILE__), __LINE__);                     \
    if constexpr (ISLOG)                                                 \
      global_logFile().log(r_server_id, omp_get_thread_num(),            \
                           basename((char *)__FILE__), __LINE__, format, \
                           ##__VA_ARGS__);                               \
  }

#define Msg_finish(format, ...)                                          \
  {                                                                      \
    if constexpr (ISCONSOLE)                                             \
      fprintf(stderr,                                                    \
              ESC_START COLOR_FINSH "[FINSH-%d T-%d]: " format           \
                                    " -> [%s:%u 行]" ESC_END "\n",       \
              r_server_id, omp_get_thread_num(), ##__VA_ARGS__,          \
              basename((char *)__FILE__), __LINE__);                     \
    if constexpr (ISLOG)                                                 \
      global_logFile().log(r_server_id, omp_get_thread_num(),            \
                           basename((char *)__FILE__), __LINE__, format, \
                           ##__VA_ARGS__);                               \
  }

#define Msg_check(format, ...)                                           \
  {                                                                      \
    if constexpr (ISCONSOLE)                                             \
      fprintf(stderr,                                                    \
              ESC_START COLOR_CHECK "[CHECK-%d T-%d]: " format           \
                                    " -> [%s:%u 行]" ESC_END "\n",       \
              r_server_id, omp_get_thread_num(), ##__VA_ARGS__,          \
              basename((char *)__FILE__), __LINE__);                     \
    if constexpr (ISLOG)                                                 \
      global_logFile().log(r_server_id, omp_get_thread_num(),            \
                           basename((char *)__FILE__), __LINE__, format, \
                           ##__VA_ARGS__);                               \
  }

#define Msg_write(format, ...)                                           \
  {                                                                      \
    if constexpr (ISCONSOLE)                                             \
      fprintf(stderr,                                                    \
              ESC_START COLOR_WRITE "[WRITE-%d T-%d]: " format           \
                                    " -> [%s:%u 行]" ESC_END "\n",       \
              r_server_id, omp_get_thread_num(), ##__VA_ARGS__,          \
              basename((char *)__FILE__), __LINE__);                     \
    if constexpr (ISLOG)                                                 \
      global_logFile().log(r_server_id, omp_get_thread_num(),            \
                           basename((char *)__FILE__), __LINE__, format, \
                           ##__VA_ARGS__);                               \
  }

#define Msg_free(format, ...)                                            \
  {                                                                      \
    if constexpr (ISCONSOLE)                                             \
      fprintf(stderr,                                                    \
              ESC_START COLOR_FREES "[FREES-%d T-%d]: " format           \
                                    " -> [%s:%u 行]" ESC_END "\n",       \
              r_server_id, omp_get_thread_num(), ##__VA_ARGS__,          \
              basename((char *)__FILE__), __LINE__);                     \
    if constexpr (ISLOG)                                                 \
      global_logFile().log(r_server_id, omp_get_thread_num(),            \
                           basename((char *)__FILE__), __LINE__, format, \
                           ##__VA_ARGS__);                               \
  }

#define Msg_logs(format, ...)                                            \
  {                                                                      \
    if constexpr (ISCONSOLE)                                             \
      fprintf(stderr,                                                    \
              ESC_START COLOR_LOG "[LOGER-%d T-%d]: " format             \
                                  " -> [%s:%u 行]" ESC_END "\n",         \
              r_server_id, omp_get_thread_num(), ##__VA_ARGS__,          \
              basename((char *)__FILE__), __LINE__);                     \
    if constexpr (ISLOG)                                                 \
      global_logFile().log(r_server_id, omp_get_thread_num(),            \
                           basename((char *)__FILE__), __LINE__, format, \
                           ##__VA_ARGS__);                               \
  }

#define Msg_major(format, ...)                                           \
  {                                                                      \
    if constexpr (ISCONSOLE)                                             \
      fprintf(stderr,                                                    \
              ESC_START COLOR_NOTICE "[FATAL-%d T-%d]: " format          \
                                     " -> [%s:%u 行]" ESC_END "\n",      \
              r_server_id, omp_get_thread_num(), ##__VA_ARGS__,          \
              basename((char *)__FILE__), __LINE__);                     \
    if constexpr (ISLOG)                                                 \
      global_logFile().log(r_server_id, omp_get_thread_num(),            \
                           basename((char *)__FILE__), __LINE__, format, \
                           ##__VA_ARGS__);                               \
  }

#define Msg_warn(format, ...)                                            \
  {                                                                      \
    if constexpr (ISCONSOLE)                                             \
      fprintf(stderr,                                                    \
              ESC_START COLOR_WARN "[WARNS-%d T-%d]: " format            \
                                   " -> [%s:%u 行]" ESC_END "\n",        \
              r_server_id, omp_get_thread_num(), ##__VA_ARGS__,          \
              basename((char *)__FILE__), __LINE__);                     \
    if constexpr (ISLOG)                                                 \
      global_logFile().log(r_server_id, omp_get_thread_num(),            \
                           basename((char *)__FILE__), __LINE__, format, \
                           ##__VA_ARGS__);                               \
  }

#define Msg_error(format, ...)                                           \
  {                                                                      \
    if constexpr (ISCONSOLE)                                             \
      fprintf(stderr,                                                    \
              ESC_START COLOR_ERROR "[ERRNB-%d T-%d]: " format           \
                                    " -> [%s:%u 行]" ESC_END "\n",       \
              r_server_id, omp_get_thread_num(), ##__VA_ARGS__,          \
              basename((char *)__FILE__), __LINE__);                     \
    if constexpr (ISLOG)                                                 \
      global_logFile().log(r_server_id, omp_get_thread_num(),            \
                           basename((char *)__FILE__), __LINE__, format, \
                           ##__VA_ARGS__);                               \
  }
#define Test_passed(format, ...)                                         \
  {                                                                      \
    if constexpr (ISCONSOLE)                                             \
      fprintf(stderr,                                                    \
              ESC_START COLOR_TEST_PASSED "※ [PASSD-%d T-%d]: " format   \
                                          " -> [%s:%u 行]" ESC_END "\n", \
              r_server_id, omp_get_thread_num(), ##__VA_ARGS__,          \
              basename((char *)__FILE__), __LINE__);                     \
    if constexpr (ISLOG)                                                 \
      global_logFile().log(r_server_id, omp_get_thread_num(),            \
                           basename((char *)__FILE__), __LINE__, format, \
                           ##__VA_ARGS__);                               \
  }
#define Test_failed(format, ...)                                         \
  {                                                                      \
    if constexpr (ISCONSOLE)                                             \
      fprintf(stderr,                                                    \
              ESC_START COLOR_TEST_FAILED "※ [FAILD-%d T-%d]: " format   \
                                          " -> [%s:%u 行]" ESC_END "\n", \
              r_server_id, omp_get_thread_num(), ##__VA_ARGS__,          \
              basename((char *)__FILE__), __LINE__);                     \
    if constexpr (ISLOG)                                                 \
      global_logFile().log(r_server_id, omp_get_thread_num(),            \
                           basename((char *)__FILE__), __LINE__, format, \
                           ##__VA_ARGS__);                               \
  }

#define Msg_node(format, ...)                                            \
  {                                                                      \
    if (r_server_id == 0)                                                \
    {                                                                    \
      if constexpr (ISCONSOLE)                                           \
        fprintf(stderr,                                                  \
                ESC_START COLOR_INFOS "[SERVE-0]: " format               \
                                      " -> [%s:%u 行]" ESC_END "\n",     \
                ##__VA_ARGS__, basename((char *)__FILE__), __LINE__);    \
    }                                                                    \
    if constexpr (ISLOG)                                                 \
      global_logFile().log(r_server_id, omp_get_thread_num(),            \
                           basename((char *)__FILE__), __LINE__, format, \
                           ##__VA_ARGS__);                               \
  }

#define GPU_info(format, ...)                                      \
  {                                                                \
    if constexpr (ISCONSOLE)                                       \
      printf(ESC_START COLOR_LOG_GPU "[GTId-%2d]: " format         \
                                     " -> [%s:%u L]" ESC_END "\n", \
             GPU_THREAD_ID, ##__VA_ARGS__, __FILE__, __LINE__);    \
  }

#define msginfo Msg_info
#define msgfinish Msg_finish
#define msgcheck Msg_check
#define msgwrite Msg_write
#define msgfree Msg_free
#define msglog Msg_logs
#define msgfatal Msg_major
#define msgwarning Msg_warn
#define msgwarn Msg_warn
#define msgerror Msg_error
#define msgnode Msg_node
#define testpass Test_passed
#define testfail Test_failed
#define gpuinfo GPU_info

/* *********************************************************************************************************
 * @description: We will update console_V3_3 to console_V4 (2024-4-19)
 *               1. we add <*_smart> ,which can set appropriate placeholders
 * automatically
 *               2. You can use it by "{}" other than "%*"
 *               3. However, it may cause small overhead (Not sure)
 * @return [*]
 * *********************************************************************************************************/
#define CONSOLE_SMART 1

#if (CONSOLE_SMART == 1)

namespace CPJ
{

constexpr const char *MY_PHR = "{}";

/* *********************************************************************************************************
 * @description: 计算自定义的占位符(MY_PHR)的数量
 * @param [string&] str format
 * @return [*]
 * *********************************************************************************************************/
inline int count_MY_PHR(const std::string &str)
{
  int count = 0;
  size_t pos = 0;
  while ((pos = str.find(MY_PHR, pos)) != std::string::npos)
  {
    count++;
    pos += std::strlen(MY_PHR);
  }
  return count;
}

/* *********************************************************************************************************
 * @description: 将自定义的占位符(MY_PHR)替换成对应到 占位符
 * @param [STR_T&] format  format
 * @param [int&] pos_start 起始查询位置(提高性能)
 * @param [T&] value 不会实际用到值，只是为了获取类型
 * @return [*] true: 找到了占位符(MY_PHR)并替换, false:
 * 没有找到占位符(MY_PHR)(不应该发生)
 * *********************************************************************************************************/
template <typename STR_T, typename T>
inline bool replacePHR(STR_T &format, int &pos_start, T &value)
{
  const char *temp = CPJ::TYPE_HOLDER<T>();
  size_t pos_cur = format.find(MY_PHR, pos_start);
  if (pos_cur != std::string::npos)
  {
    format.replace(pos_cur, std::strlen(MY_PHR), temp);
    return true;
  }
  return false;
}

/* *********************************************************************************************************
 * @description: 折叠表达式展开(c++17)
 *               其余的展开方式参考资料:
 * https://blog.csdn.net/albertsh/article/details/123978539 逻辑与操作符 &&
 * 保证参数全部展开 (用","也可以)
 * @param [string&] format
 * @param [int&] pos 起始查询位置(提高性能)
 * @param [Args&&...] args
 * @return [*]
 * *********************************************************************************************************/
template <typename... Args>
inline void parse_args(std::string &format, int &pos, Args &&...args)
{
  (replacePHR(format, pos, args) && ...);
}

/* *********************************************************************************************************
 * @description: 检查参数个数与自定义的占位符(MY_PHR)个数是否相等
 * @param [string] &format
 * @param [Args] &
 * @return [*] true: 相等, false: 不等
 * *********************************************************************************************************/
template <typename... Args>
inline bool checkPHRnum(const std::string &format, Args &&...args)
{
  return (count_MY_PHR(format) == sizeof...(args));
}

}  // namespace CPJ

//! ----------------------------------
#define assert_msg_smart(condition, format, ...)                           \
  if (true)                                                                \
  {                                                                        \
    if (__builtin_expect(!(condition), 0))                                 \
    {                                                                      \
      std::string format_str = ESC_START COLOR_ERROR                       \
          "<ERROR-%d>: " format " -> T[%u] [%s:%u 行]\n" ESC_END;          \
      assert_msg(CPJ::checkPHRnum(format_str, __VA_ARGS__),                \
                 "PlaceHolder can not match args");                        \
      int pos = 0;                                                         \
      CPJ::parse_args(format_str, pos, __VA_ARGS__);                       \
      fprintf(stderr, format_str.c_str(), r_server_id, ##__VA_ARGS__,      \
              omp_get_thread_num(), basename((char *)__FILE__), __LINE__); \
      fflush(stderr);                                                      \
      if constexpr (ISLOG == 1)                                            \
      {                                                                    \
        global_logFile().myFlush();                                        \
        global_logFile().myClose();                                        \
      }                                                                    \
      exit(EXIT_FAILURE);                                                  \
    }                                                                      \
  }

#define assert_msg_clear_smart(condition, format, ...)                     \
  if (true)                                                                \
  {                                                                        \
    if (__builtin_expect(!(condition), 0))                                 \
    {                                                                      \
      std::string format_str = ESC_START COLOR_ERROR                       \
          "<ERGO-%d>: " format " -> T[%u] [%s:%u 行]\n" ESC_END;           \
      assert_msg(CPJ::checkPHRnum(format_str, __VA_ARGS__),                \
                 "PlaceHolder can not match args");                        \
      int pos = 0;                                                         \
      CPJ::parse_args(format_str, pos, __VA_ARGS__);                       \
      fprintf(stderr, format_str.c_str(), r_server_id, ##__VA_ARGS__,      \
              omp_get_thread_num(), basename((char *)__FILE__), __LINE__); \
      fflush(stderr);                                                      \
      if constexpr (ISLOG == 1)                                            \
      {                                                                    \
        global_logFile().myFlush();                                        \
        global_logFile().myClose();                                        \
      }                                                                    \
      goto clear;                                                          \
    }                                                                      \
  }

#define Msg_info_smart(format, ...)                                            \
  {                                                                            \
    if constexpr (ISCONSOLE || ISLOG)                                          \
    {                                                                          \
      std::string format_str = format;                                         \
      assert_msg(CPJ::checkPHRnum(format_str, ##__VA_ARGS__),                  \
                 "PlaceHolder can not match args");                            \
      int pos = 0;                                                             \
      CPJ::parse_args(format_str, pos, ##__VA_ARGS__);                         \
      if constexpr (ISCONSOLE)                                                 \
      {                                                                        \
        std::string console_str =                                              \
            ESC_START COLOR_INFOS "[INFOS-%d T-%d]: " + format_str +           \
            " -> [%s:%u 行]\n" ESC_END;                                        \
        fprintf(stderr, console_str.c_str(), r_server_id,                      \
                omp_get_thread_num(), ##__VA_ARGS__,                           \
                basename((char *)__FILE__), __LINE__);                         \
      }                                                                        \
      if constexpr (ISLOG)                                                     \
        global_logFile().log(r_server_id, omp_get_thread_num(),                \
                             basename((char *)__FILE__), __LINE__, format_str, \
                             ##__VA_ARGS__);                                   \
    }                                                                          \
  }
   // T-%d
#define msginfo_n(format, ...)                                              \
  {                                                                         \
    if constexpr (ISCONSOLE || ISLOG)                                       \
    {                                                                       \
      std::string format_str = format;                                      \
      assert_msg(CPJ::checkPHRnum(format_str, ##__VA_ARGS__),               \
                 "PlaceHolder can not match args");                         \
      int pos = 0;                                                          \
      CPJ::parse_args(format_str, pos, ##__VA_ARGS__);                      \
      if constexpr (ISCONSOLE)                                              \
      {                                                                     \
        std::string console_str =                                           \
            ESC_START COLOR_INFOS "" + format_str + " " ESC_END;            \
        fprintf(stderr, console_str.c_str(), ##__VA_ARGS__);                \
      }                                                                     \
      if constexpr (ISLOG) global_logFile().log(format_str, ##__VA_ARGS__); \
    }                                                                       \
  }

#define Msg_finish_smart(format, ...)                                          \
  {                                                                            \
    if constexpr (ISCONSOLE || ISLOG)                                          \
    {                                                                          \
      std::string format_str = format;                                         \
      assert_msg(CPJ::checkPHRnum(format_str, __VA_ARGS__),                    \
                 "PlaceHolder can not match args");                            \
      int pos = 0;                                                             \
      CPJ::parse_args(format_str, pos, __VA_ARGS__);                           \
      if constexpr (ISCONSOLE)                                                 \
      {                                                                        \
        std::string console_str =                                              \
            ESC_START COLOR_FINSH "[FINSH-%d T-%d]: " + format_str +           \
            " -> [%s:%u 行]\n" ESC_END;                                        \
        fprintf(stderr, console_str.c_str(), r_server_id,                      \
                omp_get_thread_num(), ##__VA_ARGS__,                           \
                basename((char *)__FILE__), __LINE__);                         \
      }                                                                        \
      if constexpr (ISLOG)                                                     \
        global_logFile().log(r_server_id, omp_get_thread_num(),                \
                             basename((char *)__FILE__), __LINE__, format_str, \
                             ##__VA_ARGS__);                                   \
    }                                                                          \
  }
/*
#define Msg_check_smart(format, ...) \
    { \
        if constexpr (ISCONSOLE || ISLOG) \
        { \
            std::string format_str = format; \
            assert_msg(CPJ::checkPHRnum(format_str, __VA_ARGS__), "PlaceHolder
can not match args");                   \
            int pos = 0; \
            CPJ::parse_args(format_str, pos, __VA_ARGS__); \
            if constexpr (ISCONSOLE) \
            { \
                std::string console_str = \
                    ESC_START COLOR_CHECK "[CHECK-%d]: " + format_str + " ->
[%s:%u 行]\n" ESC_END;                    \
                fprintf(stderr, console_str.c_str(), r_server_id, ##__VA_ARGS__,
basename((char *)__FILE__), __LINE__); \
            } \
            if constexpr (ISLOG) \
                global_logFile().log(r_server_id, omp_get_thread_num(),
basename((char *)__FILE__), __LINE__,           \
                                     format_str, ##__VA_ARGS__); \
        } \
    }
*/
#define Msg_check_smart(format, ...)                                           \
  {                                                                            \
    if constexpr (ISCONSOLE || ISLOG)                                          \
    {                                                                          \
      std::string format_str = format;                                         \
      assert_msg(CPJ::checkPHRnum(format_str, __VA_ARGS__),                    \
                 "PlaceHolder can not match args");                            \
      int pos = 0;                                                             \
      CPJ::parse_args(format_str, pos, __VA_ARGS__);                           \
      if constexpr (ISCONSOLE)                                                 \
      {                                                                        \
        std::string console_str =                                              \
            ESC_START COLOR_CHECK "[CHECK-%d T-%d]: " + format_str +           \
            " -> [%s:%u 行]\n" ESC_END;                                        \
        fprintf(stderr, console_str.c_str(), r_server_id,                      \
                omp_get_thread_num(), ##__VA_ARGS__,                           \
                basename((char *)__FILE__), __LINE__);                         \
      }                                                                        \
      if constexpr (ISLOG)                                                     \
        global_logFile().log(r_server_id, omp_get_thread_num(),                \
                             basename((char *)__FILE__), __LINE__, format_str, \
                             ##__VA_ARGS__);                                   \
    }                                                                          \
  }

#define Msg_write_smart(format, ...)                                           \
  {                                                                            \
    if constexpr (ISCONSOLE || ISLOG)                                          \
    {                                                                          \
      std::string format_str = format;                                         \
      assert_msg(CPJ::checkPHRnum(format_str, __VA_ARGS__),                    \
                 "PlaceHolder can not match args");                            \
      int pos = 0;                                                             \
      CPJ::parse_args(format_str, pos, __VA_ARGS__);                           \
      if constexpr (ISCONSOLE)                                                 \
      {                                                                        \
        std::string console_str =                                              \
            ESC_START COLOR_WRITE "[WRITE-%d T-%d]: " + format_str +           \
            " -> [%s:%u 行]\n" ESC_END;                                        \
        fprintf(stderr, console_str.c_str(), r_server_id,                      \
                omp_get_thread_num(), ##__VA_ARGS__,                           \
                basename((char *)__FILE__), __LINE__);                         \
      }                                                                        \
      if constexpr (ISLOG)                                                     \
        global_logFile().log(r_server_id, omp_get_thread_num(),                \
                             basename((char *)__FILE__), __LINE__, format_str, \
                             ##__VA_ARGS__);                                   \
    }                                                                          \
  }

#define Msg_free_smart(format, ...)                                            \
  {                                                                            \
    if constexpr (ISCONSOLE || ISLOG)                                          \
    {                                                                          \
      std::string format_str = format;                                         \
      assert_msg(CPJ::checkPHRnum(format_str, __VA_ARGS__),                    \
                 "PlaceHolder can not match args");                            \
      int pos = 0;                                                             \
      CPJ::parse_args(format_str, pos, __VA_ARGS__);                           \
      if constexpr (ISCONSOLE)                                                 \
      {                                                                        \
        std::string console_str =                                              \
            ESC_START COLOR_FREES "[FREES-%d T-%d]: " + format_str +           \
            " -> [%s:%u 行]\n" ESC_END;                                        \
        fprintf(stderr, console_str.c_str(), r_server_id,                      \
                omp_get_thread_num(), ##__VA_ARGS__,                           \
                basename((char *)__FILE__), __LINE__);                         \
      }                                                                        \
      if constexpr (ISLOG)                                                     \
        global_logFile().log(r_server_id, omp_get_thread_num(),                \
                             basename((char *)__FILE__), __LINE__, format_str, \
                             ##__VA_ARGS__);                                   \
    }                                                                          \
  }

#define Msg_logs_smart(format, ...)                                            \
  {                                                                            \
    if constexpr (ISCONSOLE || ISLOG)                                          \
    {                                                                          \
      std::string format_str = format;                                         \
      assert_msg(CPJ::checkPHRnum(format_str, __VA_ARGS__),                    \
                 "PlaceHolder can not match args");                            \
      int pos = 0;                                                             \
      CPJ::parse_args(format_str, pos, __VA_ARGS__);                           \
      if constexpr (ISCONSOLE)                                                 \
      {                                                                        \
        std::string console_str =                                              \
            ESC_START COLOR_LOG "[LOGER-%d T-%d]: " + format_str +             \
            " -> [%s:%u 行]\n" ESC_END;                                        \
        fprintf(stderr, console_str.c_str(), r_server_id,                      \
                omp_get_thread_num(), ##__VA_ARGS__,                           \
                basename((char *)__FILE__), __LINE__);                         \
      }                                                                        \
      if constexpr (ISLOG)                                                     \
        global_logFile().log(r_server_id, omp_get_thread_num(),                \
                             basename((char *)__FILE__), __LINE__, format_str, \
                             ##__VA_ARGS__);                                   \
    }                                                                          \
  }

#define Msg_major_smart(format, ...)                                           \
  {                                                                            \
    if constexpr (ISCONSOLE || ISLOG)                                          \
    {                                                                          \
      std::string format_str = format;                                         \
      assert_msg(CPJ::checkPHRnum(format_str, __VA_ARGS__),                    \
                 "PlaceHolder can not match args");                            \
      int pos = 0;                                                             \
      CPJ::parse_args(format_str, pos, __VA_ARGS__);                           \
      if constexpr (ISCONSOLE)                                                 \
      {                                                                        \
        std::string console_str =                                              \
            ESC_START COLOR_NOTICE "[FATAL-%d T-%d]: " + format_str +          \
            " -> [%s:%u 行]\n" ESC_END;                                        \
        fprintf(stderr, console_str.c_str(), r_server_id,                      \
                omp_get_thread_num(), ##__VA_ARGS__,                           \
                basename((char *)__FILE__), __LINE__);                         \
      }                                                                        \
      if constexpr (ISLOG)                                                     \
        global_logFile().log(r_server_id, omp_get_thread_num(),                \
                             basename((char *)__FILE__), __LINE__, format_str, \
                             ##__VA_ARGS__);                                   \
    }                                                                          \
  }

#define Msg_warn_smart(format, ...)                                            \
  {                                                                            \
    if constexpr (ISCONSOLE || ISLOG)                                          \
    {                                                                          \
      std::string format_str = format;                                         \
      assert_msg(CPJ::checkPHRnum(format_str, __VA_ARGS__),                    \
                 "PlaceHolder can not match args");                            \
      int pos = 0;                                                             \
      CPJ::parse_args(format_str, pos, __VA_ARGS__);                           \
      if constexpr (ISCONSOLE)                                                 \
      {                                                                        \
        std::string console_str =                                              \
            ESC_START COLOR_WARN "[WARNS-%d T-%d]: " + format_str +            \
            " -> [%s:%u 行]\n" ESC_END;                                        \
        fprintf(stderr, console_str.c_str(), r_server_id,                      \
                omp_get_thread_num(), ##__VA_ARGS__,                           \
                basename((char *)__FILE__), __LINE__);                         \
      }                                                                        \
      if constexpr (ISLOG)                                                     \
        global_logFile().log(r_server_id, omp_get_thread_num(),                \
                             basename((char *)__FILE__), __LINE__, format_str, \
                             ##__VA_ARGS__);                                   \
    }                                                                          \
  }

#define Msg_error_smart(format, ...)                                           \
  {                                                                            \
    if constexpr (ISCONSOLE || ISLOG)                                          \
    {                                                                          \
      std::string format_str = format;                                         \
      assert_msg(CPJ::checkPHRnum(format_str, __VA_ARGS__),                    \
                 "PlaceHolder can not match args");                            \
      int pos = 0;                                                             \
      CPJ::parse_args(format_str, pos, __VA_ARGS__);                           \
      if constexpr (ISCONSOLE)                                                 \
      {                                                                        \
        std::string console_str =                                              \
            ESC_START COLOR_ERROR "[ERRNB-%d T-%d]: " + format_str +           \
            " -> [%s:%u 行]\n" ESC_END;                                        \
        fprintf(stderr, console_str.c_str(), r_server_id,                      \
                omp_get_thread_num(), ##__VA_ARGS__,                           \
                basename((char *)__FILE__), __LINE__);                         \
      }                                                                        \
      if constexpr (ISLOG)                                                     \
        global_logFile().log(r_server_id, omp_get_thread_num(),                \
                             basename((char *)__FILE__), __LINE__, format_str, \
                             ##__VA_ARGS__);                                   \
    }                                                                          \
  }

#define Test_passed_smart(format, ...)                                         \
  {                                                                            \
    if constexpr (ISCONSOLE || ISLOG)                                          \
    {                                                                          \
      std::string format_str = format;                                         \
      assert_msg(CPJ::checkPHRnum(format_str, __VA_ARGS__),                    \
                 "PlaceHolder can not match args");                            \
      int pos = 0;                                                             \
      CPJ::parse_args(format_str, pos, __VA_ARGS__);                           \
      if constexpr (ISCONSOLE)                                                 \
      {                                                                        \
        std::string console_str =                                              \
            ESC_START COLOR_TEST_PASSED "※ [PASSD-%d T-%d]: " + format_str +   \
            " -> [%s:%u 行]\n" ESC_END;                                        \
        fprintf(stderr, console_str.c_str(), r_server_id,                      \
                omp_get_thread_num(), ##__VA_ARGS__,                           \
                basename((char *)__FILE__), __LINE__);                         \
      }                                                                        \
      if constexpr (ISLOG)                                                     \
        global_logFile().log(r_server_id, omp_get_thread_num(),                \
                             basename((char *)__FILE__), __LINE__, format_str, \
                             ##__VA_ARGS__);                                   \
    }                                                                          \
  }

#define Test_failed_smart(format, ...)                                         \
  {                                                                            \
    if constexpr (ISCONSOLE || ISLOG)                                          \
    {                                                                          \
      std::string format_str = format;                                         \
      assert_msg(CPJ::checkPHRnum(format_str, __VA_ARGS__),                    \
                 "PlaceHolder can not match args");                            \
      int pos = 0;                                                             \
      CPJ::parse_args(format_str, pos, __VA_ARGS__);                           \
      if constexpr (ISCONSOLE)                                                 \
      {                                                                        \
        std::string console_str =                                              \
            ESC_START COLOR_TEST_FAILED "※ [FAILD-%d T-%d]: " + format_str +   \
            " -> [%s:%u 行]\n" ESC_END;                                        \
        fprintf(stderr, console_str.c_str(), r_server_id,                      \
                omp_get_thread_num(), ##__VA_ARGS__,                           \
                basename((char *)__FILE__), __LINE__);                         \
      }                                                                        \
      if constexpr (ISLOG)                                                     \
        global_logFile().log(r_server_id, omp_get_thread_num(),                \
                             basename((char *)__FILE__), __LINE__, format_str, \
                             ##__VA_ARGS__);                                   \
    }                                                                          \
  }

#define Msg_node_smart(format, ...)                                            \
  {                                                                            \
    if constexpr (ISCONSOLE || ISLOG)                                          \
    {                                                                          \
      std::string format_str = format;                                         \
      assert_msg(CPJ::checkPHRnum(format_str, __VA_ARGS__),                    \
                 "PlaceHolder can not match args");                            \
      int pos = 0;                                                             \
      CPJ::parse_args(format_str, pos, __VA_ARGS__);                           \
      if constexpr (ISCONSOLE)                                                 \
      {                                                                        \
        std::string console_str =                                              \
            ESC_START COLOR_INFOS "[SERVE-0]: " + format_str +                 \
            " -> [%s:%u 行]\n" ESC_END;                                        \
        fprintf(stderr, console_str.c_str(), r_server_id, ##__VA_ARGS__,       \
                basename((char *)__FILE__), __LINE__);                         \
      }                                                                        \
      if constexpr (ISLOG)                                                     \
        global_logFile().log(r_server_id, omp_get_thread_num(),                \
                             basename((char *)__FILE__), __LINE__, format_str, \
                             ##__VA_ARGS__);                                   \
    }                                                                          \
  }

#define msginfo_s Msg_info_smart
#define msgfinish_s Msg_finish_smart
#define msgcheck_s Msg_check_smart
#define msgwrite_s Msg_write_smart
#define msgfree_s Msg_free_smart
#define msglog_s Msg_logs_smart
#define msgfatal_s Msg_major_smart
#define msgwarning_s Msg_warn_smart
#define msgwarn_s Msg_warn_smart
#define msgerror_s Msg_error_smart
#define msgnode_s Msg_node_smart
#define testpass_s Test_passed_smart
#define testfail_s Test_failed_smart
#define gpuinfo_s GPU_info_smart
#define astmsg_s assert_msg_smart
#define astmsgclr_s assert_msg_clear_smart

#endif

// #include "Basic/Time/time_cpj.hpp"
// void test_console()
// {
//     int a = 1;
//     double b = 3.14;
//     float c = 6.12;
//     uint32_t d = 5;
//     int64_t e = -100;
//     uint64_t f = 100;
//     const char g = 'c';
//     const char* h = "hello";
//     std::string i = "world";

//     CPJ::Timer t;
//     t.start();
//     printf("a = %d, b = %.2lf, c = %.2f, d = %u, e = %ld, f = %zu, g = %c, h
//     = %s, i = %s\n", a, b, c, d, e, f, g, h, i.c_str()); printf("a = %d, b =
//     %.2lf, c = %.2f, d = %u, e = %ld, f = %zu, g = %c, h = %s, i = %s\n", a,
//     b, c, d, e, f, g, h, i.c_str()); printf("a = %d, b = %.2lf, c = %.2f, d =
//     %u, e = %ld, f = %zu, g = %c, h = %s, i = %s\n", a, b, c, d, e, f, g, h,
//     i.c_str()); printf("a = %d, b = %.2lf, c = %.2f, d = %u, e = %ld, f =
//     %zu, g = %c, h = %s, i = %s\n", a, b, c, d, e, f, g, h, i.c_str());
//     printf("a = %d, b = %.2lf, c = %.2f, d = %u, e = %ld, f = %zu, g = %c, h
//     = %s, i = %s\n", a, b, c, d, e, f, g, h, i.c_str()); printf("[printf].
//     Used Time: %s", t.get_time_str().c_str());

//     t.start();
//     fprintf(stderr, "a = %d, b = %.2lf, c = %.2f, d = %u, e = %ld, f = %zu, g
//     = %c, h = %s, i = %s\n", a, b, c, d, e, f, g, h, i.c_str());
//     fprintf(stderr, "a = %d, b = %.2lf, c = %.2f, d = %u, e = %ld, f = %zu, g
//     = %c, h = %s, i = %s\n", a, b, c, d, e, f, g, h, i.c_str());
//     fprintf(stderr, "a = %d, b = %.2lf, c = %.2f, d = %u, e = %ld, f = %zu, g
//     = %c, h = %s, i = %s\n", a, b, c, d, e, f, g, h, i.c_str());
//     fprintf(stderr, "a = %d, b = %.2lf, c = %.2f, d = %u, e = %ld, f = %zu, g
//     = %c, h = %s, i = %s\n", a, b, c, d, e, f, g, h, i.c_str());
//     fprintf(stderr, "a = %d, b = %.2lf, c = %.2f, d = %u, e = %ld, f = %zu, g
//     = %c, h = %s, i = %s\n", a, b, c, d, e, f, g, h, i.c_str());
//     printf("[fprintf]. Used Time: %s", t.get_time_str().c_str());

//     t.start();
//     Msg_info("a = %d, b = %.2lf, c = %.2f, d = %u, e = %ld, f = %zu, g = %c,
//     h = %s, i = %s", a, b, c, d, e, f, g, h, i.c_str()); Msg_check("a = %d, b
//     = %.2lf, c = %.2f, d = %u, e = %ld, f = %zu, g = %c, h = %s, i = %s", a,
//     b, c, d, e, f, g, h, i.c_str()); Msg_warn("a = %d, b = %.2lf, c = %.2f, d
//     = %u, e = %ld, f = %zu, g = %c, h = %s, i = %s", a, b, c, d, e, f, g, h,
//     i.c_str()); Msg_major("a = %d, b = %.2lf, c = %.2f, d = %u, e = %ld, f =
//     %zu, g = %c, h = %s, i = %s", a, b, c, d, e, f, g, h, i.c_str());
//     Msg_error("a = %d, b = %.2lf, c = %.2f, d = %u, e = %ld, f = %zu, g = %c,
//     h = %s, i = %s", a, b, c, d, e, f, g, h, i.c_str()); printf("[msg]. Used
//     Time: %s", t.get_time_str().c_str());

//     t.start();
//     Msg_info_smart("a = {}, b = {}, c = {}, d = {}, e = {}, f = {}, g = {}, h
//     = {}, i = {}", a, b, c, d, e, f, g, h, i.c_str()); Msg_check_smart("a =
//     {}, b = {}, c = {}, d = {}, e = {}, f = {}, g = {}, h = {}, i = {}", a,
//     b, c, d, e, f, g, h, i.c_str()); Msg_warn_smart("a = {}, b = {}, c = {},
//     d = {}, e = {}, f = {}, g = {}, h = {}, i = {}", a, b, c, d, e, f, g, h,
//     i.c_str()); Msg_major_smart("a = {}, b = {}, c = {}, d = {}, e = {}, f =
//     {}, g = {}, h =
//     {}, i = {}", a, b, c, d, e, f, g, h, i.c_str()); Msg_error_smart("a = {},
//     b = {}, c = {}, d = {}, e = {}, f = {}, g = {}, h = {}, i = {}", a, b, c,
//     d, e, f, g, h, i.c_str()); printf("[smart]. Used Time: %s",
//     t.get_time_str().c_str());
// }