#include "mcp-client.hpp"

#include "whisper.h"
#include "common-whisper.h"

#include <cassert>

void pretty_print_json(const json & j) {
    printf("%s\n", j.dump(2).c_str());
}

void assert_initialized_response(const json & init_response) {
    assert(init_response.contains("id"));
    assert(init_response["id"] == 1);

    assert(init_response.contains("jsonrpc"));
    assert(init_response["jsonrpc"] == "2.0");

    assert(init_response.contains("result"));

    // Check the result object
    json result = init_response["result"];

    // Check capabilities
    assert(result.contains("capabilities"));
    assert(result["capabilities"].contains("tools"));
    assert(result["capabilities"]["tools"].is_object());

    // Check protocol version
    assert(result.contains("protocolVersion"));
    assert(result["protocolVersion"] == "2024-11-05");

    // Check server info
    assert(result.contains("serverInfo"));
    assert(result["serverInfo"].contains("name"));
    assert(result["serverInfo"]["name"] == "whisper-mcp-server");
    assert(result["serverInfo"].contains("version"));
    assert(result["serverInfo"]["version"] == "1.0.0");
}

int main() {
    std::string server_bin = "../../build/bin/whisper-mcp-server";
    std::vector<std::string> args = {
        "--model", "../../models/ggml-base.en.bin"
    };
    mcp::Client client;

    assert(client.start_server(server_bin, args));

    assert(client.wait_for_server_ready(2000));

    client.read_server_logs();

    json init_response = client.initialize("mcp-demo-client", "1.0.0");
    pretty_print_json(init_response);
    assert_initialized_response(init_response);

    return 0;
}
