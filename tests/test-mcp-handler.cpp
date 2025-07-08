#include "mcp-client.hpp"

#include "whisper.h"
#include "common-whisper.h"

#ifdef NDEBUG
#undef NDEBUG
#endif

#include <cassert>

int main() {
    mcp::Client client;

    std::string server_bin = "build/bin/whisper-mcp-server";
    assert(client.start_server(server_bin));
    assert(client.wait_for_server_ready(2000));

    json init_response = client.initialize("mcp-demo-client", "1.0.0");

    return 0;
}
