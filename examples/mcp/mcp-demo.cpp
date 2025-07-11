#include "stdio-client.hpp"

#include <string>
#include <iostream>


void print_separator(const std::string & title) {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(50, '=') << std::endl;
}

void pretty_print_json(const json & j) {
    std::cout << j.dump(2) << std::endl;
}

int main(int argc, char ** argv) {
    std::string server_command = "build/bin/whisper-mcp-server";

    if (argc > 1) {
        server_command = argv[1];
    }

    std::cout << "Starting MCP Demo" << std::endl;
    std::cout << "Server command: " << server_command << std::endl;

    try {
        mcp::StdioClient client;

        // Start the server
        print_separator("STARTING SERVER");
        if (!client.start_server(server_command)) {
            std::cerr << "Failed to start server" << std::endl;
            return 1;
        }

        if (!client.wait_for_server_ready(2000)) {
            std::cerr << "Server failed to start within timeout" << std::endl;
            return 1;
        }

        client.read_server_logs();

        // Initialize
        print_separator("INITIALIZING");
        json init_response = client.initialize("mcp-demo-client", "1.0.0");
        std::cout << "Initialize response:" << std::endl;
        pretty_print_json(init_response);

        if (init_response.contains("error")) {
            std::cerr << "Initialization failed!" << std::endl;
            return 1;
        }

        // Send initialized notification
        print_separator("SENDING INITIALIZED NOTIFICATION");
        client.send_initialized();
        client.read_server_logs();

        // List tools
        print_separator("LISTING TOOLS");
        json tools_response = client.list_tools();
        std::cout << "Tools list response:" << std::endl;
        pretty_print_json(tools_response);

        // Call transcribe tool
        print_separator("CALLING TRANSCRIBE TOOL");
        json transcribe_args = {
            {"file", "samples/jfk.wav"}
        };

        json transcribe_response = client.call_tool("transcribe", transcribe_args);
        std::cout << "Transcribe response:" << std::endl;
        pretty_print_json(transcribe_response);

        // Call model info tool
        print_separator("CALLING MODEL INFO TOOL");
        json model_info_response = client.call_tool("model_info", json::object());
        std::cout << "Model info response:" << std::endl;
        pretty_print_json(model_info_response);

        // Final logs
        print_separator("FINAL SERVER LOGS");
        client.read_server_logs();

        print_separator("DEMO COMPLETED SUCCESSFULLY");

    } catch (const std::exception & e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
