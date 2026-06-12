# Slang -> MSL compilation helper for the Metal backend.
#
# Produces .metal (MSL source) files from .slang sources using
#   slangc -target metal
# The Metal backend compiles these at runtime via newLibraryWithSource:.
# (Pre-compilation to .metallib via xcrun requires the full Xcode.app and
# is a future improvement over this development-time path.)
#
# Expects these variables to be set before use:
#   SLANGC                    - path to the slangc compiler
#   NGAPI_SHADER_OUTPUT_DIR   - directory the .metal files are written to
#   NGAPI_SHADER_INCLUDE_DIR  - include search path passed to slangc (-I)
#
# compile_metal_shader(SOURCE <path> STAGE <stage> OUTPUT <rel.metal>
#                      [ENTRY <name>...] [NUMTHREADS "X Y Z"]
#                      [EXTRA_DEPENDS ...])
#   SOURCE        - .slang path, absolute or relative to the calling CMakeLists
#   STAGE         - shader stage (compute, vertex, fragment, ...)
#   OUTPUT        - output .metal path relative to NGAPI_SHADER_OUTPUT_DIR
#   ENTRY         - entry point(s); defaults to "main". When multiple entries
#                   are specified, each gets its own .metal file named
#                   <OUTPUT_STEM>_<entry>.metal so that every kernel is always
#                   at [[buffer(0)]] (multi-entry Slang MSL assigns buffer(N)
#                   by position order).
#   NUMTHREADS    - optional "X Y Z" threadgroup size embedded as a comment
#                   "// NGAPI_THREADS X Y Z" at the top of each output file.
#                   The Metal backend reads this at pipeline creation to set
#                   the correct dispatchThreadgroups threadsPerThreadgroup.
#   EXTRA_DEPENDS - additional files that should trigger a recompile
#
# Appends produced output paths to NGAPI_METAL_SHADER_OUTPUTS in the caller's scope.

if(NOT SLANGC)
    find_program(SLANGC slangc
        HINTS "${VULKAN_SDK}/Bin" "${VULKAN_SDK}/bin" "$ENV{VULKAN_SDK}/Bin" "$ENV{VULKAN_SDK}/bin")
endif()

if(NOT PYTHON3_EXECUTABLE)
    find_program(PYTHON3_EXECUTABLE python3)
    if(NOT PYTHON3_EXECUTABLE)
        message(FATAL_ERROR "python3 not found; required for Metal shader post-processing (RewriteMetalHeaps.py)")
    endif()
endif()

function(compile_metal_shader)
    cmake_parse_arguments(S "" "SOURCE;STAGE;OUTPUT;NUMTHREADS" "ENTRY;EXTRA_DEPENDS" ${ARGN})

    if(NOT SLANGC)
        message(FATAL_ERROR "compile_metal_shader: slangc not found; put it on PATH or set VULKAN_SDK")
    endif()

    if(NOT S_STAGE)
        set(S_STAGE "compute")
    endif()

    if(NOT S_ENTRY)
        set(S_ENTRY "main")
    endif()

    if(NOT IS_ABSOLUTE "${S_SOURCE}")
        set(S_SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/${S_SOURCE}")
    endif()

    # Derive stem and extension from OUTPUT so we can build per-entry filenames.
    get_filename_component(OUT_STEM "${S_OUTPUT}" NAME_WE)
    get_filename_component(OUT_DIR_REL "${S_OUTPUT}" DIRECTORY)

    list(LENGTH S_ENTRY ENTRY_COUNT)

    foreach(entry IN LISTS S_ENTRY)
        # Single-entry compilation: each kernel gets its own file, ensuring
        # EntryPointParams is always at [[buffer(0)]].
        # Strip a leading underscore from the entry name for the filename
        # (_add -> Tensor_add.metal, not Tensor__add.metal).
        string(REGEX REPLACE "^_" "" entry_stem "${entry}")
        if(ENTRY_COUNT GREATER 1)
            set(OUT_FILE "${NGAPI_SHADER_OUTPUT_DIR}/${OUT_DIR_REL}/${OUT_STEM}_${entry_stem}.metal")
        else()
            set(OUT_FILE "${NGAPI_SHADER_OUTPUT_DIR}/${S_OUTPUT}")
        endif()

        get_filename_component(OUT_ABS_DIR "${OUT_FILE}" DIRECTORY)

        # Build the optional numthreads-comment prepend command.
        set(_PREPEND_CMD "")
        if(S_NUMTHREADS)
            set(_PREPEND_CMD
                COMMAND ${CMAKE_COMMAND}
                    "-DOUT_FILE=${OUT_FILE}"
                    "-DCOMMENT=// NGAPI_THREADS ${S_NUMTHREADS}"
                    -P "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/PrependMetalComment.cmake")
        endif()

        add_custom_command(
            OUTPUT "${OUT_FILE}"
            COMMAND ${CMAKE_COMMAND} -E make_directory "${OUT_ABS_DIR}"
            COMMAND "${SLANGC}" "${S_SOURCE}"
                    -target metal
                    -entry ${entry} -stage ${S_STAGE}
                    -I "${NGAPI_SHADER_INCLUDE_DIR}"
                    -warnings-disable 39001
                    -o "${OUT_FILE}"
            # Slang emits `EntryPointParams_N constant*` for kernel parameters, but
            # Metal's runtime compiler rejects dereferencing a device* loaded from
            # constant memory. Rewrite to `device EntryPointParams_N*` so the
            # pointer chain stays in device address space throughout.
            COMMAND sed -i "" "s/\\(EntryPointParams_[0-9][0-9]*\\) constant\\*/device \\1*/g" "${OUT_FILE}"
            # Rewrite Tier-2 multi-flexible-array structs and unbounded kernel
            # parameters so the MSL compiles with newLibraryWithSource: at runtime.
            COMMAND "${PYTHON3_EXECUTABLE}"
                "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/RewriteMetalHeaps.py"
                "${OUT_FILE}"
            ${_PREPEND_CMD}
            # The post-processing scripts are inputs too: editing the rewriter
            # must regenerate every output.
            DEPENDS "${S_SOURCE}" ${S_EXTRA_DEPENDS}
                    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/RewriteMetalHeaps.py"
                    "${CMAKE_CURRENT_FUNCTION_LIST_DIR}/PrependMetalComment.cmake"
            COMMENT "Compiling Metal shader ${entry} -> ${OUT_FILE}"
            VERBATIM)

        list(APPEND LOCAL_OUTPUTS "${OUT_FILE}")
    endforeach()

    list(APPEND NGAPI_METAL_SHADER_OUTPUTS ${LOCAL_OUTPUTS})
    set(NGAPI_METAL_SHADER_OUTPUTS "${NGAPI_METAL_SHADER_OUTPUTS}" PARENT_SCOPE)
endfunction()
