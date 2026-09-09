#!/usr/bin/env bash

export OPENAI_API_BASE_URL=http://127.0.0.1:30000/v1
export OPENAI_API_KEY=EMPTY
export ENABLE_WEB_SEARCH=true
export WEB_SEARCH_ENGINE=duckduckgo
export WEB_SEARCH_RESULT_COUNT=5
export WEB_SEARCH_CONCURRENT_REQUESTS=10
export WEB_SEARCH_ENABLED_BY_DEFAULT=true

uvx --with ddgs open-webui serve
