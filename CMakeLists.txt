cmake_minimum_required(VERSION 3.16)

project(nnr_workspace)

if(MSVC)
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL" CACHE STRING "" FORCE)
endif()

option(NNR_ENABLE_TRACE "Enable Chrome trace emitter (ui.perfetto.dev)" ON)

add_subdirectory(src)
#add_subdirectory(tests)

# Applied after subdirectories so it survives set_directory_properties in src/
if(NNR_ENABLE_TRACE)
    target_compile_definitions(nnr PUBLIC NNR_ENABLE_TRACE)
endif()
