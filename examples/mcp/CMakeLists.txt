set(TARGET whisper-mcp-server)
add_executable(${TARGET} mcp-server.cpp stdio-transport.cpp whisper-mcp-handler.cpp)
target_include_directories(${TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set(CLIENT_TARGET mcp_client)
add_executable(${CLIENT_TARGET} mcp-client.cpp)

include(DefaultTargetOptions)

target_link_libraries(${TARGET} PRIVATE common json_cpp whisper ${CMAKE_THREAD_LIBS_INIT})

# mcp_client only needs json_cpp and threading not whisper.
target_link_libraries(${CLIENT_TARGET} PRIVATE json_cpp ${CMAKE_THREAD_LIBS_INIT})

install(TARGETS ${TARGET} ${CLIENT_TARGET} RUNTIME)
