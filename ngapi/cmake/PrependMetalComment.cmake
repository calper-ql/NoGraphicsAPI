# Prepends a single comment line to a Metal shader file.
# Called via: cmake -DOUT_FILE=<path> -DCOMMENT="..." -P PrependMetalComment.cmake
file(READ "${OUT_FILE}" CONTENT)
file(WRITE "${OUT_FILE}" "${COMMENT}\n${CONTENT}")
