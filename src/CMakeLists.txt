cmake_minimum_required(VERSION 3.16)

project(nnr)

if(MSVC)
    add_compile_options(/utf-8)
endif()

include_directories(
	./
	./format/onnx
	./format/tflite
	${CMAKE_SOURCE_DIR}/third_party
)

file(GLOB_RECURSE SRC_FILES ${PROJECT_SOURCE_DIR} *.h *.hpp *.c *.cpp)
list(FILTER SRC_FILES EXCLUDE REGEX "${PROJECT_SOURCE_DIR}/build/.*")
list(FILTER SRC_FILES EXCLUDE REGEX "${CMAKE_SOURCE_DIR}/third_party/.*")

add_library(nnr
	${SRC_FILES}
)

target_include_directories(nnr PUBLIC ${CMAKE_SOURCE_DIR}/third_party)

set_directory_properties(PROPERTIES COMPILE_DEFINITIONS "")

if(WIN32)
    target_compile_definitions(nnr PRIVATE NOMINMAX)
endif()

option(NNR_NO_THREAD_POOL "Disable the thread pool (single-threaded execution)" OFF)
option(NNR_USE_OPENMP "Use OpenMP for threading instead of custom thread pool" OFF)
option(NNR_USE_POOL_V2 "Use adaptive dispatch thread pool v2 (default; set OFF for v1 escape hatch)" ON)
option(NNR_ENABLE_PROFILER "Enable instrumentation profiler (zero overhead when OFF)" OFF)
option(NNR_USE_XBYAK "Enable Xbyak JIT code generation (x64 only)" ON)
option(NNR_PROFILE_REORDERS "Dump NHWC/BLOCKED reorder count+bytes+time on exit" OFF)

if(NNR_ENABLE_PROFILER)
    target_compile_definitions(nnr PUBLIC NNR_ENABLE_PROFILER)
endif()

if(NNR_PROFILE_REORDERS)
    target_compile_definitions(nnr PRIVATE NNR_PROFILE_REORDERS)
endif()

if(NNR_USE_XBYAK)
    target_include_directories(nnr PRIVATE ${CMAKE_SOURCE_DIR}/third_party/xbyak)
    target_compile_definitions(nnr PUBLIC NNR_USE_XBYAK)
endif()

if(NNR_NO_THREAD_POOL)
    target_compile_definitions(nnr PUBLIC NNR_NO_THREAD_POOL)
elseif(NNR_USE_OPENMP)
    find_package(OpenMP REQUIRED)
    target_compile_definitions(nnr PUBLIC NNR_USE_OPENMP)
    target_link_libraries(nnr PUBLIC OpenMP::OpenMP_CXX)
elseif(NNR_USE_POOL_V2)
    target_compile_definitions(nnr PUBLIC NNR_USE_POOL_V2)
endif()

# xbyak_aarch64: JIT assembler for ARM64 NEON micro-kernels
message(STATUS "CMAKE_SYSTEM_PROCESSOR = ${CMAKE_SYSTEM_PROCESSOR}")
message(STATUS "CMAKE_GENERATOR_PLATFORM = ${CMAKE_GENERATOR_PLATFORM}")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64" OR CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64")
    message(STATUS "Enabling xbyak_aarch64 JIT support")
    set(XBYAK_AARCH64_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../third_party/xbyak_aarch64")
    target_include_directories(nnr PUBLIC ${XBYAK_AARCH64_DIR})
    target_include_directories(nnr PRIVATE ${XBYAK_AARCH64_DIR}/xbyak_aarch64 ${XBYAK_AARCH64_DIR}/src)
    target_sources(nnr PRIVATE
        ${XBYAK_AARCH64_DIR}/src/xbyak_aarch64_impl.cpp
        ${XBYAK_AARCH64_DIR}/src/util_impl.cpp
    )
    target_compile_definitions(nnr PUBLIC NNR_USE_XBYAK_AARCH64)
endif()

set_property(TARGET nnr PROPERTY CXX_STANDARD 23)

if(MSVC)
	add_definitions(-D_CRT_SECURE_NO_WARNINGS)
	target_compile_options(nnr PRIVATE /MP)
	target_compile_options(nnr PRIVATE /Zc:preprocessor)
	if(CMAKE_SYSTEM_PROCESSOR MATCHES "ARM64|aarch64" OR CMAKE_GENERATOR_PLATFORM STREQUAL "ARM64")
		# armv8.6 enables dotprod (SDOT/UDOT), fp16 arith, i8mm (SMMLA), bf16 (BFDOT) intrinsics.
		# Runtime CPU-feature gating still protects execution on older ARM chips.
		target_compile_options(nnr PUBLIC /arch:armv8.6)
	else()
		target_compile_options(nnr PRIVATE /arch:AVX2)
	endif()
	target_link_libraries(nnr PUBLIC synchronization)
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64")
	# Pi 5 is the ARM execution floor: ARMv8.2-A + dotprod (SDOT/UDOT) + fp16 arithmetic.
	# We also declare +i8mm at compile time so kernels that explicitly use SMMLA/UMMLA
	# intrinsics can compile; those kernels MUST be runtime-gated via has_neon_i8mm()
	# so the resulting binary never executes i8mm instructions on Pi 5. The compiler
	# does not spontaneously auto-vectorize scalar code with SMMLA (it's a 2×2×8 matrix
	# instruction, no common scalar pattern matches it), so this flag is safe for Pi 5.
	# +fp16fml enables `vfmlalq_{low,high,lane_low,lane_high}_f16` (FP16 widening FMAs).
	# Cortex-A76 (Pi 5) and both Oryon generations include FHM/FP16FML; safe by default.
	target_compile_options(nnr PUBLIC -march=armv8.2-a+dotprod+fp16+fp16fml+i8mm)
endif()
