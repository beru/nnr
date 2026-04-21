# WebGPU backend — Dawn integration, WGSL stamping, nnr link.
# Included from src/CMakeLists.txt when NNR_ENABLE_WEBGPU is ON.
# Expects `nnr` target to already exist in the parent scope.

set(DAWN_DIR "${CMAKE_SOURCE_DIR}/third_party/dawn")
if(NOT EXISTS "${DAWN_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "NNR_ENABLE_WEBGPU=ON but third_party/dawn is empty. Run: git submodule update --init --recursive third_party/dawn")
endif()

# Auto-fetch Dawn's transitive deps (abseil, spirv-tools, tint, jinja2, ...).
# Requires Python 3 on PATH. First configure downloads ~200MB.
set(DAWN_FETCH_DEPENDENCIES ON CACHE BOOL "" FORCE)
set(DAWN_BUILD_SAMPLES      OFF CACHE BOOL "" FORCE)
set(DAWN_BUILD_TESTS        OFF CACHE BOOL "" FORCE)
set(DAWN_BUILD_BENCHMARKS   OFF CACHE BOOL "" FORCE)
set(DAWN_BUILD_NODE_BINDINGS OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_TESTS        OFF CACHE BOOL "" FORCE)
set(TINT_BUILD_CMD_TOOLS    OFF CACHE BOOL "" FORCE)
add_subdirectory(${DAWN_DIR} ${CMAKE_BINARY_DIR}/dawn EXCLUDE_FROM_ALL)

# Dawn's HEADER_ONLY libs put generated headers in INTERFACE_SOURCES, which
# CMake tries to validate as source files at Generate time. Strip them —
# consumers only need the include dirs, not the files as sources.
foreach(tgt dawn_headers dawncpp_headers)
    if(TARGET ${tgt})
        set_property(TARGET ${tgt} PROPERTY INTERFACE_SOURCES "")
    endif()
endforeach()

target_link_libraries(nnr PUBLIC webgpu_cpp dawn::dawn_proc dawn::dawn_native)
target_compile_definitions(nnr PUBLIC NNR_ENABLE_WEBGPU)

# ----------------------------------------------------------------------------
# Stamp every wgsl/*.wgsl file into a single generated header with one entry
# per shader, keyed by the file's stem:
#     inline constexpr const char* matmul = R"(…contents…)";
#
# Add a new shader by dropping a .wgsl file into wgsl/ — no CMake edit needed.
# ----------------------------------------------------------------------------
file(GLOB WGSL_SOURCES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/wgsl/*.wgsl")
set(NNR_WGSL_DECLS "")
foreach(f ${WGSL_SOURCES})
    get_filename_component(stem "${f}" NAME_WE)
    file(READ "${f}" _src)
    string(APPEND NNR_WGSL_DECLS
        "inline constexpr const char* ${stem} = R\"__NNR_WGSL__(\n${_src})__NNR_WGSL__\";\n\n")
    # GLOB's CONFIGURE_DEPENDS only triggers on add/remove, not content edits.
    # Adding each file here makes `file(READ ...)` re-run on content change.
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${f}")
endforeach()
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/wgsl/shaders.h.in"
    "${CMAKE_BINARY_DIR}/gen/webgpu/shaders.h"
    @ONLY
)
target_include_directories(nnr PRIVATE "${CMAKE_BINARY_DIR}/gen/webgpu")
