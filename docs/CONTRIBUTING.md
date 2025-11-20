# Contributing to the ML Bootcamp Project

Thank you for your interest in contributing to this project! This document provides guidelines for contributors.

## Getting Started

1. Fork the repository on GitHub
2. Clone your fork locally
3. Create a new branch for your contribution
4. Make your changes
5. Submit a pull request

## Development Workflow

### Setting Up Your Development Environment

1. Clone the repository:
```bash
git clone https://github.com/YOUR_USERNAME/ml-bootcamp.git
cd ml-bootcamp
```

2. Create a virtual environment:
```bash
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Making Changes

1. Create a new branch:
```bash
git checkout -b feature/your-feature-name
```

2. Make your changes following the project structure and coding standards.

3. Before committing, run any tests to ensure your changes don't break existing functionality:
```bash
pytest -q
```

4. Commit your changes with a clear, concise commit message:
```bash
git add .
git commit -m "Add feature: brief description of your changes"
```

## Project Structure Guidelines

- **scripts/**: All Python modules and utilities
- **notebooks/**: Jupyter notebooks organized by workflow step
- **data/**: Data storage (actual data files excluded from git)
- **docs/**: Documentation and guides
- **assets/**: Models, figures, and other build artifacts

## Code Style and Standards

- Follow PEP 8 standards for Python code
- Use clear, descriptive variable and function names
- Add docstrings to all functions and classes
- Keep imports organized (standard library first, then third-party, then local)
- Keep code focused and avoid overly complex functions

## Documentation

- Update README.md if you change the project structure significantly
- Update relevant documentation in the docs/ folder
- Add comments to complex sections of code
- Consider adding example notebooks if you add features

## Testing

- Write tests for new functionality
- Ensure all tests pass before submitting a pull request
- Test both typical and edge cases
- Focus on testing critical functionality like data loading and model prediction

## Submitting a Pull Request

1. Push your changes to your fork:
```bash
git push origin feature/your-feature-name
```

2. Go to the GitHub repository
3. Click "New Pull Request"
4. Select your feature branch
5. Provide a clear title and description of your changes
6. Address any reviewer comments

## Types of Contributions

- **Bug fixes**: Identify and fix errors in existing code
- **New features**: Add new functionality, models, or analysis methods
- **Documentation**: Improve README, code comments, or tutorials
- **Refactoring**: Improve code structure without changing functionality
- **Performance**: Optimize existing code for better performance

## When to Implement

If you're learning through this bootcamp:

1. First, run the existing pipeline and notebooks
2. Understand the current code structure
3. Start with small improvements:
   - Fix any bugs you find
   - Add better comments/documentation
   - Improve evaluation metrics
4. Later, consider:
   - Adding new models
   - Implementing additional feature engineering techniques
   - Expanding the web API

## Questions

If you have questions about contributing:
1. Check the existing code and documentation
2. Look at similar examples in the codebase
3. Create an issue for discussion if needed
4. Be specific about what you're trying to accomplish

Thank you for contributing to this ML Bootcamp project!