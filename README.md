---
title: OpenAI to Flowith Converter
emoji: 🔄
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
env:
  - FLOWITH_AUTH_TOKEN
  - API_KEY # Add this line
secrets:
  - FLOWITH_AUTH_TOKEN
  - API_KEY # Add this line
---

# OpenAI to Flowith Converter

This project provides a simple API endpoint that accepts OpenAI-compatible chat completion requests and forwards them to the Flowith API, translating the request and response formats as needed. It's designed to be run easily using Docker, both locally and on Hugging Face Spaces.

## Setup

1.  **Environment Variables:** Create a `.env` file in the project root by copying the example: `cp .env.example .env` (or manually create it).
2.  **Flowith Token:** Open the `.env` file and replace the placeholder with your actual Flowith authorization token:
    ```
    FLOWITH_AUTH_TOKEN=your_actual_token_here
    ```
3.  **Model Mappings (Optional):** If you need to use different Flowith models or map OpenAI model names differently, you can update the [`models.json`](models.json) file.
4.  **API Key (Optional):** By default, the API uses the key `123456`. If you want to use a different key, set the `API_KEY` environment variable in your `.env` file (for local runs) or as a secret named `API_KEY` in Hugging Face Spaces.

## Running Locally (Docker)

To build and run the service locally using Docker Compose:

```bash
docker-compose up --build
```

The API will then be accessible at [`http://localhost:8099/v1/chat/completions`](http://localhost:8099/v1/chat/completions).

## Running on Hugging Face Spaces

This repository is configured for deployment on Hugging Face Spaces using Docker.

1.  Create a new Space on Hugging Face, selecting "Docker" as the SDK.
2.  Link this repository to your Space.
3.  Navigate to your Space's "Settings" page.
4.  Go to the "Secrets" section.
5.  Add a new secret with the name `FLOWITH_AUTH_TOKEN` and paste your actual Flowith authorization token as the value. The application will automatically read this secret.

The Space will build the Docker image and start the service. The API endpoint will be available at your Space's URL (e.g., `https://your-username-your-space-name.hf.space/v1/chat/completions`).

## API Endpoint

*   **URL:** `/v1/chat/completions`
*   **Method:** `POST`
*   **Request Body:** Send a JSON payload conforming to the OpenAI Chat Completions API schema (e.g., specifying `model`, `messages`, `stream`, etc.). The `model` field should correspond to a key in [`models.json`](models.json).
*   **Authentication:** Requests must include an `Authorization` header with your API key. Use the format `Bearer your_api_key`. For example, if using the default key, the header would be `Authorization: Bearer 123456`.
*   **Response:** The API will return either a standard JSON response or a server-sent event stream, mimicking the OpenAI API behavior based on the `stream` parameter in the request.