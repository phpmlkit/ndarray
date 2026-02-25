# Contributing to NDArray PHP

Thank you for your interest in contributing to NDArray PHP! This document provides guidelines and instructions for contributing.

## Code of Conduct

Be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

1. Check if the bug has already been reported
2. Create a new issue with:
   - Clear description of the bug
   - Steps to reproduce
   - Expected vs actual behavior
   - PHP version and platform
   - Code example if applicable

### Suggesting Features

1. Check if the feature has already been suggested
2. Create a new issue describing:
   - The feature you'd like to see
   - Why it would be useful
   - Any implementation ideas

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests (`composer test`)
5. Run static analysis (`composer lint`)
6. Format code (`composer cs:fix`)
7. Commit with clear messages
8. Push to your fork
9. Open a Pull Request

## Development Setup

```bash
# Clone repository
git clone https://github.com/phpmlkit/ndarray.git
cd ndarray

# Install PHP dependencies
composer install

# Install Node dependencies (for docs)
npm install

# Run tests
composer test

# Run static analysis
composer lint

# Format code
composer cs:fix
```

## Code Style

- Follow PSR-12 coding standards
- Use type hints for all parameters and return types
- Add PHPDoc comments for public methods
- Write clear, descriptive variable names
- Keep methods focused and small

## Testing

- Add tests for new features
- Ensure all tests pass before submitting PR
- Aim for high test coverage
- Test edge cases and error conditions

## Documentation

- Update documentation for new features
- Add examples to API reference
- Update relevant guide pages
- Keep NumPy compatibility notes current

## Rust Code

If modifying Rust code:
- Follow Rust best practices
- Add tests for new FFI functions
- Ensure memory safety
- Update FFI bindings in `src/FFI/Bindings.php`

## Commit Messages

- Use present tense ("Add feature" not "Added feature")
- Use imperative mood ("Move cursor to..." not "Moves cursor to...")
- Limit first line to 72 characters
- Reference issues and PRs where appropriate

## Questions?

Feel free to open an issue for questions or join discussions.

Thank you for contributing!
