#include "mcp-client.hpp"
#include <iostream>
#include <sstream>
#include <thread>
#include <chrono>
#include <stdexcept>
#include <unistd.h>
#include <sys/wait.h>
#include <signal.h>
#include <fcntl.h>

namespace mcp {

Client::Client()
    : server_pid_(-1), server_stdin_(nullptr), server_stdout_(nullptr), server_stderr_(nullptr)
    ,request_id_counter_(0) , server_running_(false) {
    stdin_pipe_[0] = stdin_pipe_[1] = -1;
    stdout_pipe_[0] = stdout_pipe_[1] = -1;
    stderr_pipe_[0] = stderr_pipe_[1] = -1;
}

Client::~Client() {
    cleanup();
}

void Client::cleanup() {
    if (server_stdin_) {
        fclose(server_stdin_);
        server_stdin_ = nullptr;
    }
    if (server_stdout_) {
        fclose(server_stdout_);
        server_stdout_ = nullptr;
    }
    if (server_stderr_) {
        fclose(server_stderr_);
        server_stderr_ = nullptr;
    }

    if (server_running_ && server_pid_ > 0) {
        kill(server_pid_, SIGTERM);
        std::this_thread::sleep_for(std::chrono::milliseconds(100));

        int status;
        if (waitpid(server_pid_, &status, WNOHANG) == 0) {
            kill(server_pid_, SIGKILL);
            waitpid(server_pid_, &status, 0);
        }
        server_running_ = false;
    }
}

bool Client::start_server(const std::string& server_command, const std::vector<std::string>& args) {
    if (server_running_) {
        return false; // Already running
    }

    // Create pipes
    if (pipe(stdin_pipe_) == -1 || pipe(stdout_pipe_) == -1 || pipe(stderr_pipe_) == -1) {
        return false;
    }

    server_pid_ = fork();
    if (server_pid_ == -1) {
        return false;
    }

    if (server_pid_ == 0) {
        // Child process - become the server
        dup2(stdin_pipe_[0], STDIN_FILENO);
        dup2(stdout_pipe_[1], STDOUT_FILENO);
        dup2(stderr_pipe_[1], STDERR_FILENO);

        // Close all pipe ends
        close(stdin_pipe_[0]); close(stdin_pipe_[1]);
        close(stdout_pipe_[0]); close(stdout_pipe_[1]);
        close(stderr_pipe_[0]); close(stderr_pipe_[1]);

        // Prepare arguments for execvp
        std::vector<char*> argv;
        argv.push_back(const_cast<char*>(server_command.c_str()));
        
        for (const auto& arg : args) {
            argv.push_back(const_cast<char*>(arg.c_str()));
        }
        argv.push_back(nullptr);

        execvp(server_command.c_str(), argv.data());
        exit(1); // exec failed
    }

    // Parent process - set up communication
    close(stdin_pipe_[0]);
    close(stdout_pipe_[1]);
    close(stderr_pipe_[1]);

    server_stdin_ = fdopen(stdin_pipe_[1], "w");
    server_stdout_ = fdopen(stdout_pipe_[0], "r");
    server_stderr_ = fdopen(stderr_pipe_[0], "r");

    if (!server_stdin_ || !server_stdout_ || !server_stderr_) {
        cleanup();
        return false;
    }

    server_running_ = true;
    return true;
}

void Client::stop_server() {
    cleanup();
}

json Client::send_request(const json & request) {
    if (!server_running_) {
        throw std::runtime_error("Server is not running");
    }

    std::string request_str = request.dump() + "\n";

    if (fputs(request_str.c_str(), server_stdin_) == EOF) {
        throw std::runtime_error("Failed to send request to server");
    }
    fflush(server_stdin_);

    // For notifications (no id), don't wait for response
    if (!request.contains("id")) {
        return json{};
    }

    // Read response
    char buffer[4096];
    if (fgets(buffer, sizeof(buffer), server_stdout_) == nullptr) {
        throw std::runtime_error("Failed to read response from server");
    }

    std::string response_str(buffer);
    if (!response_str.empty() && response_str.back() == '\n') {
        response_str.pop_back();
    }

    return json::parse(response_str);
}

void Client::read_server_logs() {
    int flags = fcntl(fileno(server_stderr_), F_GETFL, 0);
    fcntl(fileno(server_stderr_), F_SETFL, flags | O_NONBLOCK);

    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), server_stderr_) != nullptr) {
        std::cout << "[SERVER LOG] " << buffer;
    }

    fcntl(fileno(server_stderr_), F_SETFL, flags);
}

json Client::initialize(const std::string & client_name, const std::string & client_version) {
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
                {"name", client_name},
                {"version", client_version}
            }}
        }}
    };

    return send_request(request);
}

void Client::send_initialized() {
    json notification = {
        {"jsonrpc", "2.0"},
        {"method", "notifications/initialized"}
    };

    send_request(notification);
}

json Client::list_tools() {
    json request = {
        {"jsonrpc", "2.0"},
        {"id", next_request_id()},
        {"method", "tools/list"}
    };

    return send_request(request);
}

json Client::call_tool(const std::string & tool_name, const json & arguments) {
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

int Client::next_request_id() {
    return ++request_id_counter_;
}

bool Client::wait_for_server_ready(int timeout_ms) {
    auto start = std::chrono::steady_clock::now();

    while (std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start).count() < timeout_ms) {

        if (server_running_) {
            // Give server a moment to fully start up
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
            return true;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    return false;
}

std::string Client::get_last_server_logs() {
    std::stringstream logs;

    int flags = fcntl(fileno(server_stderr_), F_GETFL, 0);
    fcntl(fileno(server_stderr_), F_SETFL, flags | O_NONBLOCK);

    char buffer[1024];
    while (fgets(buffer, sizeof(buffer), server_stderr_) != nullptr) {
        logs << buffer;
    }

    fcntl(fileno(server_stderr_), F_SETFL, flags);
    return logs.str();
}

} // namespace mcp
