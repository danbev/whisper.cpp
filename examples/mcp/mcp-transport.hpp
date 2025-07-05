#pragma once

#include "json.hpp"

using json = nlohmann::ordered_json;

class MCPTransport {
public:
    virtual ~MCPTransport() = default;
    virtual void send_response(const json & response) = 0;
};
