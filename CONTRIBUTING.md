# Contributing to AutoML Distributed Platform

Thank you for your interest in contributing to the AutoML Distributed Platform! This document provides guidelines for contributing to the project.

## Table of Contents

1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Setup](#development-setup)
4. [Code Style and Standards](#code-style-and-standards)
5. [Testing](#testing)
6. [Pull Request Process](#pull-request-process)
7. [Issue Reporting](#issue-reporting)
8. [Component Guidelines](#component-guidelines)

## Code of Conduct

This project adheres to a code of conduct that promotes a welcoming and inclusive environment for all contributors. Please be respectful and constructive in all interactions.

## Getting Started

### Prerequisites

- Python 3.9+
- Docker and Docker Compose
- Node.js 16+ (for frontend development)
- kubectl (for Kubernetes deployment)
- Git

### Fork and Clone

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/AgentML.git
   cd AgentML
   ```

## Development Setup

### Quick Start

```bash
# Copy environment configuration
cp .env.example .env

# Install dependencies
make install

# Start development environment
make dev-setup

# Run the application
make dev
```

### Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Start infrastructure services
docker-compose up -d

# Initialize the platform
python main.py init

# Run the application
python main.py server --reload
```

### Frontend Development

```bash
cd frontend
npm install
npm start
```

## Code Style and Standards

### Python Code Style

- Follow PEP 8 guidelines
- Use type hints for function parameters and return values
- Maximum line length: 88 characters (Black formatter)
- Use descriptive variable and function names

### Code Formatting

We use the following tools for code formatting and linting:

```bash
# Format code
make format

# Check code style
make lint
```

### Required Tools

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking

### Documentation

- Use docstrings for all classes and functions
- Follow Google-style docstring format
- Include type information in docstrings
- Add inline comments for complex logic

Example:
```python
def train_model(
    dataset_id: str, 
    model_type: ModelType, 
    hyperparameters: Dict[str, Any]
) -> str:
    """Train a machine learning model.
    
    Args:
        dataset_id: Unique identifier for the dataset
        model_type: Type of model to train
        hyperparameters: Model hyperparameters
        
    Returns:
        The trained model identifier
        
    Raises:
        ValueError: If dataset_id is not found
        ModelTrainingError: If training fails
    """
```

## Testing

### Test Structure

```
tests/
├── test_ml_pipeline.py      # ML pipeline tests
├── test_distributed_sim.py  # Distributed simulation tests
├── test_chatbot.py         # Chatbot tests
├── test_integration.py     # Integration tests
└── conftest.py            # Test fixtures
```

### Running Tests

```bash
# Run all tests
make test

# Run specific component tests
make test-ml
make test-distributed
make test-chatbot

# Run with coverage
pytest --cov=. --cov-report=html
```

### Test Guidelines

- Write tests for all new features
- Maintain test coverage above 80%
- Use descriptive test names
- Test both success and failure scenarios
- Use fixtures for common test data

## Pull Request Process

### Before Submitting

1. Ensure your code follows the style guidelines
2. Run all tests and ensure they pass
3. Update documentation if needed
4. Add tests for new features
5. Update CHANGELOG.md if applicable

### PR Checklist

- [ ] Code follows style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] Commit messages are descriptive
- [ ] PR description explains the changes

### Review Process

1. Submit your pull request
2. Automated checks will run (CI/CD)
3. Code review by maintainers
4. Address any feedback
5. Merge after approval

## Issue Reporting

### Bug Reports

When reporting bugs, please include:

- Clear description of the issue
- Steps to reproduce
- Expected vs actual behavior
- Environment details (OS, Python version, etc.)
- Relevant logs or error messages

### Feature Requests

For feature requests, please provide:

- Clear description of the feature
- Use case and motivation
- Proposed implementation approach
- Any relevant examples or mockups

### Issue Templates

Use the provided issue templates when available:

- Bug Report
- Feature Request
- Documentation Update
- Performance Issue

## Component Guidelines

### ML Pipeline Component

- Follow MLflow conventions for experiment tracking
- Use appropriate model serialization formats
- Implement proper error handling and logging
- Support distributed training when applicable

### Distributed Simulation Component

- Implement Raft consensus algorithm correctly
- Provide clear visualization of cluster state
- Support network partition simulation
- Include comprehensive logging of consensus events

### Chatbot Component

- Use appropriate NLP libraries and models
- Implement proper intent classification
- Support contextual conversations
- Include confidence scoring for responses

### Frontend Component

- Use modern React patterns and hooks
- Implement responsive design
- Follow accessibility guidelines
- Include proper error handling

## Architecture Guidelines

### API Design

- Follow RESTful conventions
- Use appropriate HTTP status codes
- Implement proper error responses
- Include API documentation

### Database Design

- Use appropriate database schemas
- Implement proper indexing
- Handle migrations correctly
- Follow data privacy guidelines

### Security

- Implement proper authentication
- Use secure communication protocols
- Validate all inputs
- Follow security best practices

## Getting Help

### Communication Channels

- GitHub Issues: For bug reports and feature requests
- GitHub Discussions: For general questions and discussions
- Documentation: Check the project README and docs

### Resources

- [Project README](README.md)
- [API Documentation](docs/api.md)
- [Architecture Overview](docs/architecture.md)
- [Deployment Guide](docs/deployment.md)

## Recognition

Contributors who make significant contributions will be recognized in:

- Project README
- Release notes
- Hall of Fame (if applicable)

Thank you for contributing to the AutoML Distributed Platform!
