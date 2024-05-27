# Listing Generation

## Overview

This project is designed to Automate the product listing. The core functionality is implemented in `pipeline.py`.

## Getting Started

### Prerequisites

Ensure you have Python installed on your machine.

### Installation

1. **Clone the Repository:**

   ```bash
   git clone <repository-url>
   cd <repository-directory>
   ```

2. **Create a Conda Environment:**

   ```bash
   conda create --name myenv python=3.8
   conda activate myenv
   ```

3. **Install Dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

### Configuration

1. **Add .env File:**

   Create a `.env` file in the root directory of the project and add your API keys for ChatGPT, SERP, and Imgur. The file should look like this:

   ```env
   gpt_api_key=your_chatgpt_api_key
   serp_api_key=your_serp_api_key
   imgur_client_id=your_imgur_client_id
   ```

### Running the Application

To run the application, execute the following command:

```bash
python pipeline.py
```
