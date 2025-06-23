#!/bin/bash

# API base URL
BASE_URL="http://localhost:5000"

# Function to make API requests
make_request() {
    local method=$1
    local endpoint=$2
    local data=${3:-""}
    
    echo -e "\n=== Testing $method $endpoint ==="
    
    if [ -z "$data" ]; then
        curl -X $method -s -o /dev/null -w "Status: %{http_code}" "$BASE_URL$endpoint"
    else
        echo "Request data: $data"
        curl -X $method -s -o /dev/null -w "Status: %{http_code}" -H "Content-Type: application/json" -d "$data" "$BASE_URL$endpoint"
    fi
    
    echo -e "\n"
}

# Test basic endpoints
echo "Testing basic endpoints..."
make_request "GET" "/"
make_request "GET" "/setup"
make_request "GET" "/config"
make_request "GET" "/timeline"
make_request "GET" "/monitor"

# Test API endpoints
echo "Testing API endpoints..."
make_request "GET" "/api/cameras"
make_request "GET" "/api/stream/url"
make_request "GET" "/api/stream/stats"
make_request "GET" "/api/layout/current"
make_request "GET" "/api/config"
make_request "GET" "/api/system/status"
make_request "GET" "/api/system/performance"
make_request "GET" "/api/timeline/events"
make_request "GET" "/api/events"

# Test with POST data
echo "Testing POST endpoints..."
make_request "POST" "/api/setup/detect-cameras" '{}'
make_request "POST" "/api/layout/recalculate" '{}'
