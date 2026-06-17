set(LIBURING_CONFIG_HOST_H ${CMAKE_CURRENT_SOURCE_DIR}/config-host.h)
set(LIBURING_COMPAT_H
  ${CMAKE_CURRENT_SOURCE_DIR}/src/include/liburing/compat.h
)
set(LIBURING_VERSION_H
  ${CMAKE_CURRENT_SOURCE_DIR}/src/include/liburing/io_uring_version.h
)

if (NOT EXISTS ${LIBURING_CONFIG_HOST_H}
    OR NOT EXISTS ${LIBURING_COMPAT_H}
    OR NOT EXISTS ${LIBURING_VERSION_H})
  execute_process(
    COMMAND ${CMAKE_COMMAND} -E env
            CC=${CMAKE_C_COMPILER}
            CXX=${CMAKE_CXX_COMPILER}
            ./configure --use-libc
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RESULT_VARIABLE LIBURING_CONFIGURE_RESULT
  )
  if (NOT LIBURING_CONFIGURE_RESULT EQUAL 0)
    message(FATAL_ERROR "failed to configure vendored liburing")
  endif()
endif()

add_library(liburing_vendored STATIC
  src/setup.c
  src/queue.c
  src/register.c
  src/syscall.c
  src/version.c
)

target_include_directories(liburing_vendored PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src/include
)

target_include_directories(liburing_vendored PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/src
)

target_compile_definitions(liburing_vendored PRIVATE
  _GNU_SOURCE
  _LARGEFILE_SOURCE
  _FILE_OFFSET_BITS=64
  LIBURING_INTERNAL
)

target_compile_options(liburing_vendored PRIVATE
  -include ${CMAKE_CURRENT_SOURCE_DIR}/config-host.h
  -Wno-unused-parameter
)
