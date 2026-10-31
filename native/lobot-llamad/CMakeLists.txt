cmake_minimum_required(VERSION 3.14)
project(lobot-llamad CXX)

set(LLAMACPP_SRC "" CACHE PATH "llama.cpp source dir (pinned by scripts/llamacpp.lock)")
set(LLAMACPP_COMMIT "unknown" CACHE STRING "pinned llama.cpp commit sha")

if (LLAMACPP_SRC STREQUAL "")
  message(FATAL_ERROR "LLAMACPP_SRC is empty; configure via scripts/build_llamacpp.sh")
endif()

set(GGML_METAL ON)
set(LLAMA_BUILD_COMMON ON)
set(LLAMA_BUILD_TESTS OFF)
set(LLAMA_BUILD_EXAMPLES OFF)
set(LLAMA_BUILD_SERVER OFF)
set(LLAMA_BUILD_TOOLS OFF)

add_subdirectory(${LLAMACPP_SRC} llamacpp)

add_executable(lobot-llamad main.cpp)
target_include_directories(lobot-llamad PRIVATE ${LLAMACPP_SRC}/common)
target_compile_definitions(lobot-llamad PRIVATE LOBOT_LLAMACPP_COMMIT="${LLAMACPP_COMMIT}")
target_link_libraries(lobot-llamad PRIVATE llama-common llama)
