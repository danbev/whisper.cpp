#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <stdexcept>
#include <sstream>
#include <thread>
#include <chrono>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>  // for fcntl

#include "json.hpp"

using json = nlohmann::json;

class MCPClient {
private:
    pid_t server_pid;
    int    stdin_pipe[2];   // pipe for writing to server's stdin
    int    stdout_pipe[2];  // pipe for reading from server's stdout
    int    stderr_pipe[2];  // pipe for reading from server's stderr
    FILE * server_stdin;
    FILE * server_stdout;
    FILE * server_stderr;
    int    request_id_counter;
    bool   server_running;
    
    void cleanup() {
        if (server_stdin) {
            fclose(server_stdin);
            server_stdin = nullptr;
        }
        if (server_stdout) {
            fclose(server_stdout);
            server_stdout = nullptr;
        }
        if (server_stderr) {
            fclose(server_stderr);
            server_stderr = nullptr;
        }
        
        if (server_running && server_pid > 0) {
            std::cout << "Terminating server process..." << std::endl;
            kill(server_pid, SIGTERM);
            
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            
            int status;
            if (waitpid(server_pid, &status, WNOHANG) == 0) {
                kill(server_pid, SIGKILL);
                waitpid(server_pid, &status, 0);
            }
            server_running = false;
        }
    }
    
public:
    MCPClient() : server_pid(-1), server_stdin(nullptr), server_stdout(nullptr), 
                  server_stderr(nullptr), request_id_counter(0), server_running(false) {
        // Initialize pipe arrays
        stdin_pipe[0]  = stdin_pipe[1]  = -1;
        stdout_pipe[0] = stdout_pipe[1] = -1;
        stderr_pipe[0] = stderr_pipe[1] = -1;
    }
    
    ~MCPClient() {
        cleanup();
    }
    
    bool start_server(const std::string& server_command) {
        // Create pipes for communication
        if (pipe(stdin_pipe) == -1 || pipe(stdout_pipe) == -1 || pipe(stderr_pipe) == -1) {
            std::cerr << "Failed to create pipes" << std::endl;
            return false;
        }
        
        server_pid = fork();
        
        if (server_pid == -1) {
            std::cerr << "Failed to fork process" << std::endl;
            return false;
        }
        
        if (server_pid == 0) {
            // Child process - set up pipes and exec server
            
            // Redirect stdin, stdout, stderr
            dup2(stdin_pipe[0], STDIN_FILENO);   // Read from parent's write end
            dup2(stdout_pipe[1], STDOUT_FILENO); // Write to parent's read end
            dup2(stderr_pipe[1], STDERR_FILENO); // Write to parent's read end
            
            // Close all pipe ends in child
            close(stdin_pipe[0]);
            close(stdin_pipe[1]);
            close(stdout_pipe[0]);
            close(stdout_pipe[1]);
            close(stderr_pipe[0]);
            close(stderr_pipe[1]);
            
            // Execute server
            execl("/bin/sh", "sh", "-c", server_command.c_str(), nullptr);
            
            // If we get here, exec failed
            std::cerr << "Failed to execute server command: " << server_command << std::endl;
            exit(1);
        }
        
        // Parent process - set up communication
        
        // Close child ends of pipes
        close(stdin_pipe[0]);
        close(stdout_pipe[1]);
        close(stderr_pipe[1]);

        std::cout << "Parent: Closed child pipe ends..." << std::endl;

        
        // Convert file descriptors to FILE streams
        server_stdin = fdopen(stdin_pipe[1], "w");
        server_stdout = fdopen(stdout_pipe[0], "r");
        server_stderr = fdopen(stderr_pipe[0], "r");

        std::cout << "Parent: Created FILE streams..." << std::endl;

        
        if (!server_stdin || !server_stdout || !server_stderr) {
            std::cerr << "Failed to create file streams" << std::endl;
            cleanup();
            return false;
        }

        std::cout << "Parent: All streams created successfully..." << std::endl;
        
        server_running = true;
        std::cout << "Server started with PID: " << server_pid << std::endl;
        
        // Give server a moment to start up
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        
        return true;
    }

    json send_request(const json& request) {
        if (!server_running) {
            throw std::runtime_error("Server is not running");
        }

        // Send request to server
        std::string request_str = request.dump() + "\n";
        std::cout << "Sending: " << request.dump() << std::endl;

        if (fputs(request_str.c_str(), server_stdin) == EOF) {
            throw std::runtime_error("Failed to send request to server");
        }
        fflush(server_stdin);

        // For notifications (no id), don't wait for response
        if (!request.contains("id")) {
            std::cout << "Notification sent (no response expected)" << std::endl;
            return json{};
        }

        int expected_id = request["id"];

        // Read response from server
        char buffer[4096];
        if (fgets(buffer, sizeof(buffer), server_stdout) == nullptr) {
            throw std::runtime_error("Failed to read response from server");
        }

        std::string response_str(buffer);
        if (!response_str.empty() && response_str.back() == '\n') {
            response_str.pop_back();
        }

        std::cout << "Received: " << response_str << std::endl;

        try {
            json response = json::parse(response_str);

            // Verify the response ID matches the request ID
            if (response.contains("id") && response["id"] != expected_id) {
                std::cerr << "Warning: Response ID mismatch. Expected: " << expected_id
                          << ", Got: " << response["id"] << std::endl;
            }

            return response;
        } catch (const json::parse_error & e) {
            throw std::runtime_error("Failed to parse JSON response: " + std::string(e.what()));
        }
    }
    
    void read_server_logs() {
        // Make stderr non-blocking
        int flags = fcntl(fileno(server_stderr), F_GETFL, 0);
        fcntl(fileno(server_stderr), F_SETFL, flags | O_NONBLOCK);

        char buffer[1024];
        while (true) {
            if (fgets(buffer, sizeof(buffer), server_stderr) == nullptr) {
                break;
            }
            std::cout << "[SERVER LOG] " << buffer;
        }

        // Reset to blocking mode
        fcntl(fileno(server_stderr), F_SETFL, flags);
    }
    
    int next_request_id() {
        return ++request_id_counter;
    }
    
    json initialize() {
        json request = {
            {"jsonrpc", "2.0"},
            {"id", next_request_id()},
            {"method", "initialize"},
            {"params", {
                {"protocolVersion", "2024-11-05"},
                {"capabilities", {
                    {"tools", json::object()}
                }},
                {"clientInfo", {
                    {"name", "whisper-mcp-client"},
                    {"version", "1.0.0"}
                }}
            }}
        };
        
        return send_request(request);
    }
    
    void send_initialized() {
        json notification = {
            {"jsonrpc", "2.0"},
            {"method", "notifications/initialized"}
        };
        
        send_request(notification);
    }
    
    json list_tools() {
        json request = {
            {"jsonrpc", "2.0"},
            {"id", next_request_id()},
            {"method", "tools/list"}
        };
        
        return send_request(request);
    }
    
    json call_tool(const std::string& tool_name, const json& arguments) {
        json request = {
            {"jsonrpc", "2.0"},
            {"id", next_request_id()},
            {"method", "tools/call"},
            {"params", {
                {"name", tool_name},
                {"arguments", arguments}
            }}
        };
        
        return send_request(request);
    }
};

void print_separator(const std::string& title) {
    std::cout << "\n" << std::string(50, '=') << std::endl;
    std::cout << title << std::endl;
    std::cout << std::string(50, '=') << std::endl;
}

void pretty_print_json(const json& j) {
    std::cout << j.dump(2) << std::endl;
}

int main(int argc, char* argv[]) {
    std::string server_command = "build/bin/whisper-mcp-server";
    
    if (argc > 1) {
        server_command = argv[1];
    }
    
    std::cout << "Starting MCP Client Demo" << std::endl;
    std::cout << "Server command: " << server_command << std::endl;
    
    try {
        MCPClient client;
        
        // Start the server
        print_separator("STARTING SERVER");
        if (!client.start_server(server_command)) {
            std::cerr << "Failed to start server" << std::endl;
            return 1;
        }
        
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        client.read_server_logs();
        
        print_separator("INITIALIZING");
        json init_response = client.initialize();
        std::cout << "Initialize response:" << std::endl;
        pretty_print_json(init_response);
        
        if (init_response.contains("error")) {
            std::cerr << "Initialization failed!" << std::endl;
            return 1;
        }
        
        print_separator("SENDING INITIALIZED NOTIFICATION");
        client.send_initialized();
        
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        client.read_server_logs();
        
        print_separator("LISTING TOOLS");
        json tools_response = client.list_tools();
        std::cout << "Tools list response:" << std::endl;
        pretty_print_json(tools_response);
        
        print_separator("CALLING TRANSCRIBE TOOL");
        json transcribe_args = {
            {"file", "samples/jfk.wav"}
        };
        
        json transcribe_response = client.call_tool("transcribe", transcribe_args);
        std::cout << "Transcribe response:" << std::endl;
        pretty_print_json(transcribe_response);
        
        print_separator("FINAL SERVER LOGS");
        client.read_server_logs();
        
        print_separator("DEMO COMPLETED SUCCESSFULLY");
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
