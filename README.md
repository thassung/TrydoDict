# TrydoDict

## Overview

Welcome to the Machine Translation Demo App! This web-based application demonstrates the basic functionality of a machine translation, allowing users to input a word or sentence in English and retrieve the generated translation in Thai.

## Dataset

The machine translation is trained with a data from [Hugging Face](https://huggingface.co/datasets/scb_mt_enth_2020) which is a large English-Thai parallel corpus, curated from news, Wikipedia articles, SMS messages, task-based dialogs, web-crawled data and government documents.

## Features

- **Input:** User can enter an English word or sentence in the provided input section.
- **Submit Button:** User clicks *submit* after typing the prompt. The app will generate the translation from the provided text.
- **Generated Translation:** A translation will be generated and shown next to the submit button. The app might give &lt;unk&gt; in the generated text if the input contains unknown word.

### Prerequisites

- Ensure you have Python installed

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/thassung/ChatABC.git
   ```

2. Install the required Python dependencies:

   ```bash
   pip install flask torch torchtext
   ```

3. Navigate to the app directoty:
   ```bash
   cd ChatABC/app
   ```

4. Start the flask application:
   ```bash
   python main.py
   ```

   You can access the application via http://localhost:8080

![Sample text generation from ChatABC](./app/templates/sample.PNG)
