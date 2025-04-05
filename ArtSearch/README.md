# ArtSearch 🔍 [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A local search system implementation using Elasticsearch for Wikipedia data indexing and retrieval.

<!-- ![Search Demo](./assets/search-demo.gif) Add actual demo file path later -->

## Table of Contents

- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
  - [Building Index](#building-index)
  - [Performing Searches](#performing-searches)
- [Configuration](#configuration)
- [Development](#development)
- [Contributing](#contributing)
- [License](#license)

## Features ✨

- Multi-language support for Wikipedia data
- Elasticsearch-powered search backend
- CLI interface for index management and queries
- Configurable search parameters

## Prerequisites 🛠️

- Python 3.12
- Java 11+ (for Elasticsearch)
- 30GB+ free disk space (for data storage)

## Installation ⚙️

### 1. Download Elasticsearch Engine

```bash
# Create data directory
mkdir -p data && cd data

# Download and extract Elasticsearch
wget -O elasticsearch-8.17.3.tar.gz \
  https://artifacts.elastic.co/downloads/elasticsearch/elasticsearch-8.17.3-linux-x86_64.tar.gz
tar zxvf elasticsearch-8.17.3.tar.gz
rm elasticsearch-8.17.3.tar.gz
cd ..
```

### 2. Download Wikipedia Data

Download specific language version of Wikipedia dataset:

```bash
# Default English dataset (November 2023)
modelscope download --dataset wikimedia/wikipedia \
  --include 20231101.en/* \
  --local_dir ./data/wikipedia
```

Example data structure:

```json
{
    "id": "1",
    "url": "https://simple.wikipedia.org/wiki/April",
    "title": "April",
    "text": "April is the fourth month..."
}
```

## Usage 🚀

### Building Index

```bash
# Build index for default language (en)
python es_wiki_build.py

# Build index for specific language (e.g., French)
python es_wiki_build.py --language fr
```

### Performing Searches

```bash
# Default search setting
python es_wiki_search.py

# Direct query execution
python es_wiki_search.py \
  --language en \
  --query "Paris 2024 Olympic Games" 
```

## Configuration ⚙️

Setting environment variables for Elasticsearch configuration:

```bash
export ELASTIC_PASSWORD="changeme"
```

<!-- Modify config.yaml for custom settings:

```yaml
elasticsearch:
  host: localhost
  port: 9200
  index_prefix: "wiki_"
languages: [en, fr, es]
batch_size: 500
``` -->

## Development 🧑💻

```bash
# Install dependencies
pip install -r requirements.txt
```

## Contributing 🤝

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create your feature branch (git checkout -b feature/amazing-feature)
3. Commit your changes (git commit -m 'Add some amazing feature')
4. Push to the branch (git push origin feature/amazing-feature)
5. Open a Pull Request

<!-- See CONTRIBUTING.md for detailed guidelines. -->

## License 📄

This project is licensed under the MIT License - see the [LICENSE](./LICENSE) file for details.

