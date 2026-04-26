cmake_minimum_required(VERSION 3.22)

# Dawn generates headers at build time; CMP0118 NEW makes GENERATED
# source-file properties visible to consuming targets across directories.
if(POLICY CMP0118)
    cmake_policy(SET CMP0118 NEW)
endif()

project(nnr_workspace)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" CACHE STRING "" FORCE)
endif()

option(NNR_ENABLE_TRACE "Enable Chrome trace emitter (ui.perfetto.dev)" ON)
option(NNR_EXPLICIT_REORDERS "T3 M1: explicit Reorder ops at layout boundaries" ON)

add_subdirectory(src)
#add_subdirectory(tests)

# Applied after subdirectories so it survives set_directory_properties in src/
if(NNR_ENABLE_TRACE)
    target_compile_definitions(nnr PUBLIC NNR_ENABLE_TRACE)
endif()
if(NNR_EXPLICIT_REORDERS)
    target_compile_definitions(nnr PUBLIC NNR_EXPLICIT_REORDERS)
endif()
