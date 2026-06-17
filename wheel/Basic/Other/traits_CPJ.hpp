#pragma once

/* *********************************************************************************************************
 * @description: 抽时间确定怎么实现 (我们想获取一个函数返回值的类型,
 * 但是重载函数会有一定的问题) 参考资料:
 *               //
 * https://stackoverflow.com/questions/40135561/decltype-for-overloaded-member-function
 *               //
 * https://stackoverflow.com/questions/22291737/why-cant-decltype-work-with-overloaded-functions
 * @return [*]
 * *********************************************************************************************************/

#include <sys/types.h>

#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>

namespace CPJ
{

class Traits
{
 public:
  template <typename T>
  static constexpr bool is_float_point()
  {
    if constexpr (std::is_floating_point_v<
                      std::remove_cv_t<std::remove_reference_t<T>>>)
      return true;
    return false;
  }

  template <typename T>
  static constexpr bool is_float_type()
  {
    if constexpr (std::is_floating_point_v<
                      std::remove_cv_t<std::remove_reference_t<T>>> &&
                  std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>,
                                 float>)
      return true;
    return false;
  }

  template <typename T>
  static constexpr bool is_double_type()
  {
    if constexpr (std::is_floating_point_v<
                      std::remove_cv_t<std::remove_reference_t<T>>> &&
                  std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>,
                                 double>)
      return true;
    return false;
  }

  template <typename T>
  static constexpr bool is_pointer()
  {
    return std::is_pointer_v<T>;
    // return std::is_pointer_v<std::remove_cv_t<std::remove_reference_t<T>>>;
  }

  template <typename T>
  static constexpr bool is_integral_point()
  {
    if constexpr (std::is_integral_v<
                      std::remove_cv_t<std::remove_reference_t<T>>>)
      return true;
    return false;
  }

  template <typename T>
  static constexpr bool is_int_type()
  {
    if constexpr (is_integral_point<T>() &&
                  std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>,
                                 int>)
      return true;
    return false;
  }

  template <typename T>
  static constexpr bool is_bool_type()
  {
    if constexpr (is_integral_point<T>() &&
                  std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>,
                                 bool>)
      return true;
    return false;
  }

  template <typename T>
  static constexpr bool is_u8_type()
  {
    if constexpr (is_integral_point<T>() &&
                  std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>,
                                 uint8_t>)
      return true;
    return false;
  }

  template <typename T>
  static constexpr bool is_u32_type()
  {
    if constexpr (is_integral_point<T>() &&
                  std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>,
                                 uint32_t>)
      return true;
    return false;
  }

  template <typename T>
  static constexpr bool is_i64_type()
  {
    if constexpr (is_integral_point<T>() &&
                  std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>,
                                 int64_t>)
      return true;
    return false;
  }

  template <typename T>
  static constexpr bool is_u64_type()
  {
    if constexpr (is_integral_point<T>() &&
                  std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>,
                                 uint64_t>)
      return true;
    return false;
  }

  template <typename T>
  static constexpr bool is_ssize_type()
  {
    if constexpr (is_integral_point<T>() &&
                  std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>,
                                 ssize_t>)
      return true;
    return false;
  }

  template <typename T>
  static constexpr bool is_char_type()
  {
    if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>,
                                 char>)
      return true;
    return false;
  }

  template <typename T>
  static constexpr bool is_chars_type()
  {
    if constexpr (is_pointer<T>())
    {
      using PlainType = std::remove_pointer_t<T>;
      if constexpr (is_char_type<PlainType>())
      {
        return true;
      }
    }
    return false;
  }

  template <typename T>
  static constexpr bool is_string_type()
  {
    if constexpr (std::is_same_v<std::remove_cv_t<std::remove_reference_t<T>>,
                                 std::string>)
      return true;
    return false;
  }

  /* *********************************************************************************************************
   * @description: 浮点类型既可以表示正数也可以表示负数, 但是不应该参与判断
   * @return [*]
   * *********************************************************************************************************/
  template <typename T>
  static constexpr bool is_signed()
  {
    if constexpr (std::numeric_limits<
                      std::remove_cv_t<std::remove_reference_t<T>>>::is_signed)
      return true;
    return false;
  }
};
}  // namespace CPJ

// template <typename Func, typename... Args>
// constexpr bool has_return_type(Func, Args&&...)
// {
//     return !std::is_same<decltype((*Func)(Args...)), void>::value;
// }

// template <typename R, typename... Args>
// constexpr bool has_return_type(R (*func)(Args&&... args))
// {
//     Msg_info("1");
//     return !std::is_void<R>::value;
// }

// template <typename R, typename... Args>
// constexpr bool has_return_type(R (*func)())
// {
//     Msg_info("2");
//     return !std::is_void<R>::value;
// }

// template <typename Func, typename... Args>
// constexpr bool has_return_type(Args&&... args)
// {
//     // 使用 decltype 和 std::declval 获取函数返回类型
//     using ReturnType =
//     decltype(std::declval<Func>()(std::forward<Args>(args)...));
//     // 返回值是否为 void
//     return !std::is_same<ReturnType, void>::value;
// }

// template <typename Func>
// constexpr bool has_return_type()
// {
//     // 使用 decltype 和 std::declval 获取函数返回类型
//     using ReturnType = decltype(std::declval<Func>()());
//     // 返回值是否为 void
//     return !std::is_same<ReturnType, void>::value;
// }

// template <typename Func, typename... Args>
// constexpr bool has_return_type()
// {
//     // 使用 decltype 和 std::declval 获取函数返回类型
//     using ReturnType =
//     decltype(std::declval<Func>()(std::declval<Args>()...));
//     // 返回值是否为 void
//     return !std::is_same<ReturnType, void>::value;
// }

// void t() { Msg_info("这是测试函数"); }
// int t(int a) { return a; }

// template <typename Func>
// void test_func()
// {
//     using ReturnType = decltype(std::declval<Func>()());
//     if constexpr (std::is_same_v<ReturnType, void>)
//     {
//         std::cout << "函数无返回值" << std::endl;
//     }
//     else
//     {
//         std::cout << "函数返回类型: " << typeid(ReturnType).name() <<
//         std::endl;
//     }
// }

// template <typename Func, typename Arg>
// void test_func()
// {
//     using ReturnType = decltype(std::declval<Func>()(std::declval<Arg>()));
//     std::cout << "函数返回类型-: " << typeid(ReturnType).name() << std::endl;
// }

// template <typename Func>
// void test_func(Func func)
// {
//     using ReturnType = decltype(func());
//     if constexpr (std::is_same_v<ReturnType, void>)
//     {
//         std::cout << "函数无返回值" << std::endl;
//     }
//     else
//     {
//         std::cout << "函数返回类型: " << typeid(ReturnType).name() <<
//         std::endl;
//     }
// }

// template <typename Func, typename Arg>
// void test_func(Func func, Arg arg)
// {
//     using ReturnType = decltype(func(arg));
//     std::cout << "函数返回类型: " << typeid(ReturnType).name() << std::endl;
// }

// template <typename Func, typename... Args>
// bool is_return_void()
// {
//     using ReturnType =
//     auto(Args...)->decltype(std::declval<Func>()(std::declval<Args>()...));
//     return !std::is_same<ReturnType, void>::value;
// }

// template <typename Func, typename... Args>
// bool is_return_void_1(Func func, Args&&... args)
// {
//     using ReturnType = decltype(func(std::forward<Args>(args)...));
//     return std::is_same<ReturnType, void>::value;
// }

// template <typename... Args>
// bool is_return_void_2(std::function<void(Args...)> func, Args&&... args)
// {
//     using ReturnType = typename
//     std::result_of<decltype(func)&(Args...)>::type; return
//     std::is_same<ReturnType, void>::value;
// }

// template <typename... Args>
// bool is_return_void_3()
// {
//     using ReturnType = std::invoke_result_t<decltype(t()), Args...>;
//     return std::is_same<ReturnType, void>::value;
// }

// template <typename R, typename... Args>
// constexpr bool has_return_type(R (*func)(Args&&... args))

// https://stackoverflow.com/questions/40135561/decltype-for-overloaded-member-function
// https://stackoverflow.com/questions/22291737/why-cant-decltype-work-with-overloaded-functions
// template <typename Func, typename... Args>
// bool ok(Func func, Args&&... args)
// {
//     using ReturnType =
//     auto(Args...)->decltype((*func)(std::declval<Args>()...)); return
//     !std::is_same<ReturnType, void>::value;
// }

// template <typename... Args>
// bool ok(Args&&... args)
// {
//     using ReturnType = auto(Args...)->decltype((t)(std::declval<Args>()...));
//     return !std::is_same<ReturnType, void>::value;
// }

// void main_traits()
// {
//     // 传递函数指针 t，不调用它
//     // test_func<decltype(t)>(); // 传递无参函数
//     // test_func<decltype(t), int>(); // 传递有参函数

//     // is_return_void(t);     // 传递无参函数
//     Msg_info("ok(10) has return: %d", ok(10)); // 传递有参函数
//     // Msg_info("ok() has return: %d", ok());
//     // is_return_void_1(t, 10);
//     // bool result1 = is_return_void_2(std::function<void()>(&t));
//     // bool result2 = is_return_void_2(std::function<int(int)>(&t), 1);

//     // is_return_void_3<int>();

//     // std::cout << "Result for t(): " << std::boolalpha << result1 <<
//     std::endl;
//     // std::cout << "Result for t(int): " << std::boolalpha << result2 <<
//     std::endl;
// }